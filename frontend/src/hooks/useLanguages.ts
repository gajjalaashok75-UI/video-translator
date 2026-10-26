/**
 * The language registry.
 *
 * Never cached in a module-level constant and never duplicated: this is the only
 * source of language support in the frontend, and it comes from `/api/languages`
 * on every mount so a backend change is reflected without a frontend deploy.
 */

import { getLanguages } from "../api/endpoints";
import type { LanguagesResponse } from "../api/types";
import { useAsync, type AsyncState } from "./useAsync";

export function useLanguages(): AsyncState<LanguagesResponse> {
  return useAsync<LanguagesResponse>((signal) => getLanguages(signal), []);
}
