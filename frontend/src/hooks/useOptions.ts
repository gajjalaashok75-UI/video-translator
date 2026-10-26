/**
 * The dropdown option lists the backend offers (`GET /api/options`).
 *
 * Every list carries an `available` flag and a `state` per item, so the UI can
 * show an option that exists but cannot be chosen yet, with the backend's own
 * reason, instead of hiding it. Nothing here filters: that decision belongs to
 * the component, which knows whether the relevant feature is on.
 */

import { getOptions } from "../api/endpoints";
import type { OptionsResponse } from "../api/types";
import { useAsync, type AsyncState } from "./useAsync";

export function useOptions(): AsyncState<OptionsResponse> {
  return useAsync<OptionsResponse>((signal) => getOptions(signal), []);
}
