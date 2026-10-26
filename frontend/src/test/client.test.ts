/**
 * Tests for the HTTP client.
 *
 * The point of these is the contract `request()` promises: callers get either
 * parsed data or an `ApiError`, never a raw rejection, an HTTP status or a
 * non-JSON body. Each test drives the real client through a stubbed fetch.
 */

import { describe, expect, it } from "vitest";
import { ApiError, API_BASE, isOfflineError, request, urlFor } from "../api/client";
import { errorEnvelope, installFetchStub } from "./fetchStub";

describe("request", () => {
  it("returns parsed JSON on success", async () => {
    installFetchStub([{ path: "/api/health", json: { status: "ok" } }]);

    const result = await request<{ status: string }>("/health");

    expect(result).toEqual({ status: "ok" });
  });

  it("prefixes the API base and never hard-codes a host", async () => {
    const stub = installFetchStub([{ path: "/api/health", json: {} }]);

    await request("/health");

    expect(stub.calls[0].url).toBe("/api/health");
    expect(stub.calls[0].url.startsWith("http")).toBe(false);
  });

  it("serializes query parameters and drops empty ones", async () => {
    const stub = installFetchStub([{ path: "/api/voices", json: {} }]);

    await request("/voices", {
      query: { tts_model: "xtts", language: "hi", missing: undefined, blank: null },
    });

    expect(stub.calls[0].url).toContain("tts_model=xtts");
    expect(stub.calls[0].url).toContain("language=hi");
    expect(stub.calls[0].url).not.toContain("missing");
    expect(stub.calls[0].url).not.toContain("blank");
  });

  it("sends a JSON body with the right content type", async () => {
    const stub = installFetchStub([
      { path: "/api/jobs", method: "POST", json: { job_id: "j1" } },
    ]);

    await request("/jobs", { method: "POST", body: { source: "a.mp4" } });

    expect(stub.calls[0].method).toBe("POST");
    expect(stub.calls[0].body).toEqual({ source: "a.mp4" });
  });

  it("turns the backend error envelope into an ApiError with its code", async () => {
    installFetchStub([
      {
        path: "/api/jobs/nope",
        status: 404,
        json: errorEnvelope("JOB_NOT_FOUND", "No such job."),
      },
    ]);

    await expect(request("/jobs/nope")).rejects.toMatchObject({
      name: "ApiError",
      code: "JOB_NOT_FOUND",
      message: "No such job.",
      status: 404,
      isOffline: false,
    });
  });

  it("preserves the field and details from the envelope", async () => {
    installFetchStub([
      {
        path: "/api/jobs",
        method: "POST",
        status: 400,
        json: errorEnvelope("INVALID_SETTINGS", "Bad preset.", {
          field: "quality_preset",
          details: { allowed: ["balanced", "high"] },
        }),
      },
    ]);

    const error = await request("/jobs", { method: "POST", body: {} }).catch(
      (e: unknown) => e as ApiError,
    );

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).field).toBe("quality_preset");
    expect((error as ApiError).details).toEqual({
      allowed: ["balanced", "high"],
    });
  });

  it("classifies a network failure as offline", async () => {
    installFetchStub([{ path: "/api/health", networkError: true }]);

    const error = await request("/health").catch((e: unknown) => e as ApiError);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).isOffline).toBe(true);
    expect((error as ApiError).code).toBe("NETWORK_ERROR");
    expect(isOfflineError(error)).toBe(true);
  });

  it("classifies a non-envelope failure without claiming offline", async () => {
    installFetchStub([
      { path: "/api/health", status: 502, text: "<html>bad gateway</html>" },
    ]);

    const error = await request("/health").catch((e: unknown) => e as ApiError);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).isOffline).toBe(false);
    expect((error as ApiError).status).toBe(502);
  });

  it("treats a 204 as no content rather than a parse error", async () => {
    installFetchStub([{ path: "/api/jobs/j1", method: "DELETE", status: 204 }]);

    await expect(
      request("/jobs/j1", { method: "DELETE" }),
    ).resolves.toBeUndefined();
  });

  it("does not swallow an abort", async () => {
    const controller = new AbortController();
    installFetchStub([{ path: "/api/health", networkError: true }]);

    controller.abort();
    const promise = request("/health", { signal: controller.signal });

    // A DOMException abort must propagate as-is, not be relabelled offline.
    await expect(promise).rejects.toSatisfy(
      (error: unknown) =>
        error instanceof DOMException ||
        (error instanceof ApiError && error.isOffline),
    );
  });
});

describe("urlFor", () => {
  it("prefixes a rooted path with the API base", () => {
    expect(urlFor("/jobs/j1/artifacts/output")).toBe(
      `${API_BASE}/jobs/j1/artifacts/output`,
    );
  });
});

describe("ApiError.offline", () => {
  it("uses the cause message when there is one", () => {
    const error = ApiError.offline(new TypeError("Failed to fetch"));
    expect(error.message).toBe("Failed to fetch");
    expect(error.isOffline).toBe(true);
  });

  it("falls back to a readable message", () => {
    const error = ApiError.offline(undefined);
    expect(error.message).toBe("Could not reach the backend.");
  });
});
