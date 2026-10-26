/**
 * Runtime environment facts for the dashboard.
 *
 * Fetched once per mount; the machine's CPU, GPU and FFmpeg do not change while
 * the page is open, so there is nothing to poll for.
 */

import { getSystem } from "../api/endpoints";
import type { SystemResponse } from "../api/types";
import { useAsync, type AsyncState } from "./useAsync";

export function useSystem(): AsyncState<SystemResponse> {
  return useAsync<SystemResponse>((signal) => getSystem(signal), []);
}
