/**
 * Optional component availability (the `/api/models` probe list).
 */

import { getModels } from "../api/endpoints";
import type { ComponentStatus } from "../api/types";
import { useAsync, type AsyncState } from "./useAsync";

export function useModels(): AsyncState<ComponentStatus[]> {
  return useAsync<ComponentStatus[]>((signal) => getModels(signal), []);
}
