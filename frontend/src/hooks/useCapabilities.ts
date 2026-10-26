/**
 * What this installation can do: backends, features, presets and (optionally)
 * the translation route matrix.
 *
 * `includeRoutes` is off by default because computing the matrix imports the
 * translation package on the server and costs seconds on the first call. Only
 * the language explorer asks for it.
 */

import { getCapabilities } from "../api/endpoints";
import type { CapabilitiesResponse } from "../api/types";
import { useAsync, type AsyncState } from "./useAsync";

export function useCapabilities(
  includeRoutes = false,
): AsyncState<CapabilitiesResponse> {
  return useAsync<CapabilitiesResponse>(
    (signal) => getCapabilities({ includeRoutes, signal }),
    [includeRoutes],
  );
}
