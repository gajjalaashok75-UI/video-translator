/**
 * The TTS voices a backend offers for a language.
 *
 * Keyed on the pair the backend's `/api/voices` requires (`tts_model` and
 * `language`), and skipped entirely until both are known — asking with an empty
 * language is a guaranteed 400. A cloning backend reports `cloning: true` and an
 * empty list, which is a real answer ("supply a reference, pick nothing"), not a
 * failure, so it is passed through rather than treated as an error.
 *
 * The hook is always called (a conditional `useAsync` would break the rules of
 * hooks); when the pair is incomplete the fetcher resolves to `null` without
 * touching the network, and `data` stays null.
 */

import { getVoices } from "../api/endpoints";
import type { VoicesResponse } from "../api/types";
import { useAsync, type AsyncState } from "./useAsync";

export function useVoices(
  ttsModel: string | null,
  language: string | null,
): AsyncState<VoicesResponse | null> {
  return useAsync<VoicesResponse | null>(
    (signal) =>
      ttsModel && language
        ? getVoices(ttsModel, language, signal)
        : Promise.resolve(null),
    [ttsModel, language],
  );
}
