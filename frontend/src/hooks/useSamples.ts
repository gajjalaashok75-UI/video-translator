/**
 * The bundled demo samples (`GET /api/samples`).
 *
 * The list is the backend's allow-list, never a directory listing, and each
 * entry carries its own `available` flag. The demo page therefore gates itself
 * on real server state rather than assuming the HGF clip is on disk.
 */

import { getSamples } from "../api/endpoints";
import type { SampleListResponse } from "../api/types";
import { useAsync, type AsyncState } from "./useAsync";

export function useSamples(): AsyncState<SampleListResponse> {
  return useAsync<SampleListResponse>((signal) => getSamples(signal), []);
}
