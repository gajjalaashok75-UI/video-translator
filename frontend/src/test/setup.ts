/**
 * Vitest setup: DOM matchers and a guaranteed-unhandled-rejection guard.
 *
 * `EventSource` and `matchMedia` do not exist in jsdom, so minimal stand-ins are
 * installed here. They are intentionally tiny: the tests that exercise the event
 * stream assert on the frames the module produces, not on a full SSE
 * implementation.
 */

import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

if (!window.matchMedia) {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
}
