"""
The job event stream: contract, replay buffer and subscriber fan-out.

Two rules shape this module.

**Structured state, not log text.** A client must be able to drive its UI from
the events alone, so every event carries a typed payload (`progress.fraction`,
`result.output_available`, …). Human-readable log lines are still published as
`log` events, but they are decoration: nothing a client needs to know lives only
in a sentence.

**Bounded memory.** A render emits a log line per stage, so an unbounded history
would grow without limit. Each job keeps a fixed-size ring of recent events;
older ones are dropped and a client that asks for one is told to `resync` from
the durable state (`GET /api/jobs/{id}`) rather than being sent nothing. The
manifest and the job log on disk remain the authoritative, complete record.

Threading: events are published from the pipeline's worker thread and consumed
by asyncio tasks in the server's event loop. A subscriber is an
`asyncio.Queue`, and the worker thread never touches it directly — it hands the
event to `loop.call_soon_threadsafe`, so all queue mutation happens on the loop
thread. That is the whole bridge between the two worlds.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.api import settings

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """
    The event vocabulary.

    `snapshot` is the current durable state, sent first on every connection so a
    client never has to call the status endpoint just to render something.
    `queued` through `cancelled` describe a job's life. `resync` means "your
    position is no longer in the buffer; re-read the job", and `heartbeat` is a
    keep-alive that deliberately does not consume an event id.
    """

    SNAPSHOT = "snapshot"
    QUEUED = "queued"
    STARTED = "started"
    PROGRESS = "progress"
    LOG = "log"
    RESULT = "result"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RESYNC = "resync"
    HEARTBEAT = "heartbeat"


@dataclass
class Event:
    """One event on the wire."""

    event_id: int
    job_id: str
    type: str
    at: float
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "job_id": self.job_id,
            "type": self.type,
            "at": self.at,
            "payload": self.payload,
        }

    def to_sse(self) -> str:
        """
        Serialise as a Server-Sent Events frame.

        `id:` carries the event id so the browser's `EventSource` sends it back
        as `Last-Event-ID` on reconnect; `event:` carries the type so a client
        can subscribe per type.
        """
        data = json.dumps(self.to_dict(), ensure_ascii=False)
        return f"id: {self.event_id}\nevent: {self.type}\ndata: {data}\n\n"


def heartbeat_frame() -> str:
    """
    A keep-alive comment.

    A comment (`:`) rather than an event, because it must not consume an event
    id: replay positions are compared against real events, and a heartbeat
    slipping into that sequence would make a reconnect believe it had missed
    something.
    """
    return ": heartbeat\n\n"


class EventBuffer:
    """
    A bounded ring of events for one job, with thread-safe publishing.

    `publish` is called from the worker thread; `subscribe`/`unsubscribe` and
    queue delivery happen on the event loop. A lock guards the ring itself, and
    a separate lock guards the subscriber set.
    """

    def __init__(self, job_id: str, loop: asyncio.AbstractEventLoop,
                 maxlen: Optional[int] = None):
        self.job_id = job_id
        self._loop = loop
        self._maxlen = maxlen or settings.EVENT_BUFFER_SIZE
        self._events: list[Event] = []
        self._next_id = 1
        self._lock = threading.Lock()
        self._subscribers: dict[int, asyncio.Queue] = {}
        self._sub_lock = threading.Lock()
        self._next_sub = 1

    # -- publishing --------------------------------------------------------
    def publish(self, event_type: EventType | str,
                payload: Optional[dict] = None) -> Event:
        """
        Record an event and fan it out to every subscriber.

        Safe to call from any thread. Returns the stored event so the caller can
        log its id.
        """
        kind = (event_type.value if isinstance(event_type, EventType)
                else str(event_type))
        with self._lock:
            event = Event(
                event_id=self._next_id,
                job_id=self.job_id,
                type=kind,
                at=time.time(),
                payload=dict(payload or {}),
            )
            self._next_id += 1
            self._events.append(event)
            if len(self._events) > self._maxlen:
                # Trim in one slice rather than pop(0) per event.
                self._events = self._events[-self._maxlen:]
        self._fan_out(event)
        return event

    def _fan_out(self, event: Event) -> None:
        with self._sub_lock:
            queues = list(self._subscribers.items())
        for token, queue in queues:
            try:
                self._loop.call_soon_threadsafe(self._offer, token, queue, event)
            except RuntimeError:
                # The loop is closed; the process is shutting down. Nothing to
                # deliver to, and this must never propagate into the worker.
                return

    def _offer(self, token: int, queue: asyncio.Queue, event: Event) -> None:
        """
        Put an event on a subscriber's queue, on the loop thread.

        A full queue means the client is not keeping up. Rather than blocking
        the worker or silently dropping an event, the oldest entry is discarded
        and a `resync` is queued in its place: the client is told its view may
        be incomplete and can re-read the job, which is strictly better than
        receiving a gap it cannot detect.
        """
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            resync = Event(
                event_id=event.event_id,
                job_id=self.job_id,
                type=EventType.RESYNC.value,
                at=time.time(),
                payload={
                    "reason": "subscriber_too_slow",
                    "message": ("Events were dropped because this client could "
                                "not keep up. Re-read the job for its current "
                                "state."),
                },
            )
            try:
                queue.put_nowait(resync)
            except asyncio.QueueFull:
                pass
            return
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    # -- subscribing -------------------------------------------------------
    def subscribe(self, maxsize: Optional[int] = None) -> tuple[int, asyncio.Queue]:
        """Register a subscriber. Returns a token and its queue."""
        queue: asyncio.Queue = asyncio.Queue(
            maxsize=maxsize or settings.SUBSCRIBER_QUEUE_SIZE)
        with self._sub_lock:
            token = self._next_sub
            self._next_sub += 1
            self._subscribers[token] = queue
        return token, queue

    def unsubscribe(self, token: int) -> None:
        with self._sub_lock:
            self._subscribers.pop(token, None)

    @property
    def subscriber_count(self) -> int:
        with self._sub_lock:
            return len(self._subscribers)

    # -- replay ------------------------------------------------------------
    def replay_after(self, last_event_id: int) -> tuple[list[Event], bool]:
        """
        Events strictly after `last_event_id`, plus whether a resync is needed.

        The second value is True when the requested position has already rotated
        out of the buffer, meaning there is a gap the client cannot see. The
        caller emits a `resync` and continues with what is available.
        """
        with self._lock:
            events = list(self._events)
            if not events:
                # Nothing buffered. If the client claims to have seen events
                # they are gone, so it must resync; if it has seen none, there
                # is simply nothing to send yet.
                return [], last_event_id > 0
            oldest = events[0].event_id
            if last_event_id < oldest - 1:
                # The client's next expected id is older than the oldest we
                # still hold, so events between are lost.
                return events, True
            return [e for e in events if e.event_id > last_event_id], False

    def snapshot(self) -> Optional[Event]:
        """The newest stored event, or None when the buffer is empty."""
        with self._lock:
            return self._events[-1] if self._events else None
