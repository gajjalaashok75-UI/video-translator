/**
 * A minimal `EventSource` stand-in.
 *
 * jsdom has no `EventSource`, and the code under test needs to drive one
 * deterministically: which frames arrive, in what order, and whether the stream
 * was closed. This records every instance so a test can emit a frame and then
 * assert what the consumer did with it. Nothing above the transport is mocked —
 * the real `openJobStream` parsing and the real `useJobStream` reducer run.
 */

import { vi } from "vitest";

type Listener = (event: MessageEvent<string>) => void;

export class FakeEventSource {
  static instances: FakeEventSource[] = [];
  readonly url: string;
  readonly listeners = new Map<string, Listener[]>();
  onopen: (() => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  closed = false;

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: Listener) {
    const existing = this.listeners.get(type) ?? [];
    existing.push(listener);
    this.listeners.set(type, existing);
  }

  /** Deliver a frame of the given SSE event type. */
  emit(type: string, data: unknown) {
    const event = { data: JSON.stringify(data) } as MessageEvent<string>;
    for (const listener of this.listeners.get(type) ?? []) {
      listener(event);
    }
  }

  close() {
    this.closed = true;
  }

  /** The most recently opened instance, or undefined before any open. */
  static latest(): FakeEventSource | undefined {
    return FakeEventSource.instances[FakeEventSource.instances.length - 1];
  }
}

/** Install the stub as the global `EventSource` for the current test. */
export function installEventSourceStub() {
  FakeEventSource.instances = [];
  vi.stubGlobal("EventSource", FakeEventSource);
  return FakeEventSource;
}
