/**
 * A minimal `XMLHttpRequest` stand-in for the upload path.
 *
 * `uploadVideo` uses XHR rather than `fetch` because upload progress is only
 * observable through XHR, so a `fetch` stub cannot exercise it. This records the
 * request (url, method, body) and answers `send` with a canned response,
 * optionally firing a progress event first so the caller's progress handling is
 * exercised rather than skipped.
 */

import { vi } from "vitest";

export interface XhrStubOptions {
  status?: number;
  json?: unknown;
  /** Fire a progress event with these byte counts before completing. */
  progress?: { loaded: number; total: number };
  /** Reject with a network error instead of responding. */
  networkError?: boolean;
}

export class FakeXhr {
  static instances: FakeXhr[] = [];
  static options: XhrStubOptions = {};

  readonly upload = {
    onprogress: null as ((event: ProgressEvent) => void) | null,
  };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;
  status = 0;
  responseText = "";
  method = "";
  url = "";
  body: unknown = undefined;
  aborted = false;
  headers: Record<string, string> = {};

  constructor() {
    FakeXhr.instances.push(this);
  }

  open(method: string, url: string) {
    this.method = method;
    this.url = url;
  }

  setRequestHeader(name: string, value: string) {
    this.headers[name] = value;
  }

  send(body?: unknown) {
    this.body = body;
    const options = FakeXhr.options;
    queueMicrotask(() => {
      if (this.aborted) return;
      if (options.networkError) {
        this.onerror?.();
        return;
      }
      if (options.progress) {
        this.upload.onprogress?.({
          lengthComputable: true,
          loaded: options.progress.loaded,
          total: options.progress.total,
        } as ProgressEvent);
      }
      this.status = options.status ?? 200;
      this.responseText =
        options.json !== undefined ? JSON.stringify(options.json) : "";
      this.onload?.();
    });
  }

  abort() {
    this.aborted = true;
    this.onabort?.();
  }

  addEventListener() {}
  removeEventListener() {}
}

/** Install the stub; returns a handle to configure the next response. */
export function installXhrStub(options: XhrStubOptions = {}) {
  FakeXhr.instances = [];
  FakeXhr.options = options;
  vi.stubGlobal("XMLHttpRequest", FakeXhr);
  return FakeXhr;
}
