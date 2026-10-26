import logging
from dataclasses import dataclass
from typing import Optional

import torch
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.utils import config
from src.translation.router import TranslationPlan, TranslationRouter
from src.translation.nllb_backend import NLLBBackend
from src.core import jobs
from src.core.cache import translation_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Non-context translation is per-line, so results can be persisted in blocks.
# A crash mid-stage then keeps every block already written instead of losing the
# whole stage. Context-aware translation is one call (see translate_segments),
# so it persists at the end of that call.
TRANSLATION_FLUSH_EVERY = 50

class LLMTranslator:
    def __init__(self, model_id="tencent/HY-MT1.5-1.8B"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine model capabilities/prompt style
        self.is_llama = "llama" in model_id.lower()
        self.is_alma = "alma" in model_id.lower()
        self.is_hymt = "hy-mt" in model_id.lower()

    def unload_model(self):
        """Unload model to free VRAM."""
        if self.model:
            logger.info(f"Unloading {self.model_id}...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("LLM Translator unloaded.")

    def load_model(self, token=None):
        if self.model:
            return
        
        logger.info(f"Loading {self.model_id} on {self.device}...")
        try:
            # Use 'token' instead of 'use_auth_token' for newer transformers versions, 
            # but 'token' is backwards compatible in many recent versions. 
            # If using very old transformers, might need check. Assuming >=4.48.0 from docker reqs.
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, token=token)
            
            # Use 4-bit quantization ONLY on CUDA
            quantization_config = None
            if self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load model
            # For CPU, we might need to be explicit about dtype to avoid OOM or Half errors
            torch_dtype = "auto"
            if self.device == "cpu":
                torch_dtype = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                device_map="auto" if self.device == "cuda" else "cpu", 
                quantization_config=quantization_config,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                token=token
            )
            
            # Ensure proper pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Critical for batched generation with LLaMA/decoder models
            self.tokenizer.padding_side = 'left'
                
            logger.info(f"{self.model_id} loaded successfully.")
            
        except Exception as e:
            if "CUDA" in str(e) and self.device == "cuda":
                logger.warning(f"CUDA Error loading LLM: {e}")
                logger.warning("Switching to CPU fallback for LLM (Warning: Slow, No Quantization)...")
                # Unload partial
                self.model = None
                self.tokenizer = None
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
                self.device = "cpu"
                # Retry
                self.load_model()
                return

            logger.error(f"Failed to load LLM: {e}")
            raise

    def get_prompt(self, text, source_lang, target_lang, context_prev=None, context_next=None,
                   speaker_id=None):
        """Generates prompt based on model type and global context."""
        from src.utils.languages import get_language_name
        
        # Convert codes to full names (e.g. "en" -> "English")
        source_name = get_language_name(source_lang)
        target_name = get_language_name(target_lang)

        # A short, bounded context block. Dialogue continuity (who is
        # speaking, what was just said, what comes next) is what keeps names,
        # pronouns and terms consistent across a scene. Kept small so it
        # cannot dominate the segment being translated.
        context_lines = []
        if context_prev:
            context_lines.append(f"Context (Previous Line): {context_prev}")
        if context_next:
            context_lines.append(f"Context (Next Line): {context_next}")
        if speaker_id:
            context_lines.append(f"Speaker: {speaker_id}")
        context_block = "\n".join(context_lines)
        
        # Standard Translation Prompt
        if self.is_llama:
            # Llama 3.1 Instruct
            sys_prompt = (
                "You are a professional video translator. Translate the user's text accurately, "
                "preserving tone and style. Do not add any explanations."
            )
            if context_block:
                sys_prompt += f"\nContext (do not translate):\n{context_block}"
                
            user_prompt = f"Translate from {source_name} to {target_name}:\n\n{text}"
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            if self.tokenizer.chat_template:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                 return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        elif self.is_alma:
            # ALMA-R (Advanced Language Model-based Translator)
            # ALMA uses specific formatting: "Translate this from {src} to {tgt}:\n{src_text}\n{tgt_text}"
            # Reference: https://github.com/fe1ixxu/ALMA
            # NOTE: ALMA expects full English names (e.g. English, French)
            prompt = f"Translate this from {source_name} to {target_name}:\n{source_name}: {text}\n{target_name}:"
            logger.info(f"ALMA Prompt: {repr(prompt)}")
            return prompt
            
        else:
            # HY-MT (Default / Fallback)
            content = f"Translate the following text from {source_name} to {target_name}:\n{text}"
            if context_block:
                content = f"Context (do not translate):\n{context_block}\n\n" + content
            messages = [{"role": "user", "content": content}]
            if self.tokenizer.chat_template:
                 return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return f"<|user|>\nTranslate from {source_name} to {target_name}:\n{text}\n<|assistant|>\n"

    def translate_batch(self, texts, source_lang_code, target_lang_code, token=None):
        if not texts: return []
        self.load_model(token=token)
        
        prompts = [self.get_prompt(t, source_lang_code, target_lang_code) for t in texts]
        
        try:
            # [Fix] Enforce truncation and max length
            self.tokenizer.model_max_length = 2048 # Explicitly set
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.device)
            
            # [Fix] Safety check for Token ID OOB (Root cause of CUDA assert in Embedding)
            vocab_size = self.model.config.vocab_size
            
            # Check for negatives
            if (inputs.input_ids < 0).any():
                logger.warning("Found negative token IDs. Clamping to 0.")
                inputs.input_ids[inputs.input_ids < 0] = 0
                
            # Check for overflow
            if (inputs.input_ids >= vocab_size).any():
                logger.warning("Found token IDs exceeding vocab size. Clamping...")
                inputs.input_ids[inputs.input_ids >= vocab_size] = self.tokenizer.unk_token_id or 0
            
            # Verify pad_token_id
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None or pad_id >= vocab_size or pad_id < 0:
                logger.warning(f"Invalid pad_token_id {pad_id}. Resetting to eos_token_id or 0.")
                pad_id = self.tokenizer.eos_token_id
                if pad_id is None or pad_id >= vocab_size:
                    pad_id = 0
                self.tokenizer.pad_token_id = pad_id
            
            with torch.no_grad():
                # [Stability Fix] Disable autocast and use greedy decoding to prevent CUDA asserts
                # caused by sampling from unstable distributions in 4-bit models on Windows.
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    do_sample=False, # Robustness: greedy decoding
                    num_beams=1,
                    pad_token_id=pad_id
                )
                torch.cuda.synchronize()
                
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Slice input from output (causal LM returns input+output)
            input_len = inputs.input_ids.shape[1]
            gen_tokens = outputs[:, input_len:]
            
            final_decoded = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            
            return [r.strip() for r in final_decoded]

        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            return texts
            
    def translate_context_aware(self, text, source_lang, target_lang, prev_text=None,
                                next_text=None, speaker_id=None, token=None):
        """Translates a single segment with surrounding dialogue context."""
        self.load_model(token=token)
        prompt = self.get_prompt(
            text, source_lang, target_lang,
            context_prev=prev_text, context_next=next_text, speaker_id=speaker_id,
        )
        
        try:
            # [Fix] Enforce truncation and max length
            self.tokenizer.model_max_length = 2048
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            
            # [Fix] Safety check
            vocab_size = self.model.config.vocab_size
            if (inputs.input_ids < 0).any(): inputs.input_ids[inputs.input_ids < 0] = 0
            if (inputs.input_ids >= vocab_size).any():
                inputs.input_ids[inputs.input_ids >= vocab_size] = self.tokenizer.unk_token_id or 0
                
            # Verify pad_token_id
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None or pad_id >= vocab_size or pad_id < 0:
                pad_id = self.tokenizer.eos_token_id or 0
                self.tokenizer.pad_token_id = pad_id

            with torch.no_grad():
                # [Stability Fix] Disable autocast and use greedy decoding to prevent CUDA asserts
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=pad_id
                )
                torch.cuda.synchronize()
             
            # Slice input from output
            input_len = inputs.input_ids.shape[1]
            gen_tokens = outputs[0][input_len:]
            decoded = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            
            # Post-process ALMA
            if self.is_alma:
                 from src.utils.languages import get_language_name
                 target_name = get_language_name(target_lang)
                 if f"{target_name}:" in decoded:
                     decoded = decoded.split(f"{target_name}:")[-1].strip()
                 
            return decoded
            
        except Exception as e:
            logger.error(f"Context translation failed: {e}")
            return text

    def translate(self, text, source_lang, target_lang):
        return self.translate_batch([text], source_lang, target_lang)[0]


@dataclass
class SegmentResult:
    """
    Outcome of translating one segment.

    `text` is the translated string (empty when the segment failed and had no
    usable fallback). `ok` is False when the backend raised or returned nothing
    usable; `error` then says why. Failures are recorded per segment so a single
    bad line never sinks a feature-length translation, and so the caller can
    retry exactly the segments that failed.
    """

    text: str
    ok: bool = True
    error: Optional[str] = None
    backend: Optional[str] = None


class Translator:
    """
    Facade the pipeline calls. Owns caching, backend lifecycle and route choice,
    and returns segments with `translated_text` plus a `translation` metadata
    block describing how each one was produced.
    """

    def __init__(self, router: Optional[TranslationRouter] = None,
                 nllb_backend: Optional[NLLBBackend] = None,
                 nllb_model_id: Optional[str] = None,
                 job_cache=None):
        # The authoritative translation cache when the pipeline runs inside a
        # durable job. It is content-addressed, versioned and written
        # atomically, so a translation survives a restart and is reused only
        # when every input that produced it is unchanged. When it is absent
        # (standalone use, tests) the small temp-file cache below is used
        # instead; the two are never active at once.
        self.job_cache = job_cache
        self.translator_cache = {}
        self.llm_translator = None
        self.nllb_backend = nllb_backend or NLLBBackend(
            nllb_model_id or config.NLLB_MODEL_ID)
        self.router = router or TranslationRouter()
        # Import centralized languages
        from src.utils import languages
        self.languages = languages
        self.cache_file = config.TEMP_DIR / "translation_cache.json"
        self.cache = self._load_cache()
        # Inspection hooks for the caller / tests.
        self.last_plan: Optional[TranslationPlan] = None
        self.last_failures: list[dict] = []

    def unload_model(self):
        """Unload any loaded translation model and clear caches."""
        if self.llm_translator:
            self.llm_translator.unload_model()
            self.llm_translator = None
        if self.nllb_backend:
            self.nllb_backend.unload()
        self.translator_cache.clear()

    def _load_cache(self):
        import json
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception: return {}
        return {}

    def _save_cache(self):
        # Atomic: a crash mid-write must not leave a half-parsed cache behind.
        try:
            jobs.atomic_write_json(self.cache_file, self.cache)
        except Exception as e:
            logger.warning(f"Failed to save translation cache: {e}")

    def get_google_translator(self, target_lang):
        # Use centralized map
        code = self.languages.get_language_code(target_lang)

        if code not in self.translator_cache:
            self.translator_cache[code] = GoogleTranslator(source='auto', target=code)
        return self.translator_cache[code], code

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------
    def _resolve_codes(self, source_lang, target_lang) -> tuple[str, str]:
        """
        Resolve UI names/codes to internal codes.

        Kept lenient on purpose: the pipeline and existing tests pass display
        names ("Spanish"), short codes ("es") and "auto". An unresolved source
        falls back to English, matching the historical contract.
        """
        if source_lang in (None, "", "auto", "Auto Detect"):
            src = "en"
        else:
            src = self.languages.get_language_code(source_lang)
        tgt = self.languages.get_language_code(target_lang)
        return src, tgt

    def _available_backends(self) -> set[str]:
        """Backends usable right now. NLLB counts only if its weights are local."""
        available = {"google", "hymt", "llama", "alma"}
        if self.nllb_backend and self.nllb_backend.is_available():
            available.add("nllb")
        return available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def translate_segments(self, segments, target_lang, model="google", source_lang="auto",
                           optimize=False, check_cancel_callback=None, hf_token=None,
                           concise="normal", target_durations=None):
        """
        Translate dialogue segments.

        :param model: requested backend ('google', 'nllb', 'hymt', 'llama',
            'alma'), or None/'auto' to let the router choose (NLLB for Indic
            pairs when it is available).
        :param optimize: thread neighbouring lines + speaker as context.
        :param concise: 'normal' | 'concise' | 'very_concise'. Recorded in the
            metadata now; shortening is deliberately not applied yet.
        :param target_durations: optional per-segment target duration in seconds,
            recorded for a later duration-aware step. Not used to shorten text.
        """
        if not segments:
            return []

        src, tgt = self._resolve_codes(source_lang, target_lang)
        requested = None if model in (None, "auto") else model

        self.last_failures = []

        # Source == target: nothing to translate. Copy through, keep metadata.
        if src == tgt:
            self.last_plan = self.router.route(src, tgt, preferred_backend=requested)
            out = []
            for seg in segments:
                s = seg.copy()
                s['translated_text'] = seg.get('text', '')
                s['source_text'] = seg.get('text', '')
                s['translation'] = self._meta(self.last_plan, concise, "passthrough", None)
                out.append(s)
            logger.info("Source == target (%s); skipping translation.", src)
            return out

        # Auto: prefer the local NLLB model when either endpoint is Indic (its
        # strength), and keep the fast online backend for other pairs. Both are
        # any-to-any, so the router can still substitute if one is unavailable.
        if requested is None:
            preferred = "nllb" if (self.languages.is_indic(src) or self.languages.is_indic(tgt)) else "google"
        else:
            preferred = requested

        available = self._available_backends()
        plan = self.router.route(src, tgt, preferred_backend=preferred, available=available)
        self.last_plan = plan
        logger.info("Translation route %s (backend=%s, pivot=%s).",
                    plan.path, plan.backend, plan.is_pivot)

        # NLLB and Google are sentence-level MT: they have no dialogue-context
        # input, so an `optimize` request is honoured only by the LLM backends.
        # Say so rather than silently dropping it.
        context_applied = optimize and plan.backend in ("hymt", "llama", "alma")
        if optimize and not context_applied:
            logger.warning(
                "Context-aware translation is only available with the LLM backends "
                "(HY-MT / Llama / ALMA); backend '%s' translates each line "
                "independently.", plan.backend)

        # Model token used in the cache key. Explicit choices keep their key so
        # the existing cache stays valid; an auto route keys on the chosen backend.
        cache_model = requested or plan.backend

        # Content-addressed lookup. A key is derived from the text and every
        # non-text input that can change the result (languages, backend,
        # optimize/concise, and the neighbouring lines when the backend uses
        # context), so a changed input produces a different key rather than a
        # stale hit. The key is computed once per segment and reused for the
        # store below, so lookup and store cannot drift apart.
        keys: list[str] = [""] * len(segments)
        results: list[Optional[dict]] = [None] * len(segments)
        pending_idx: list[int] = []
        for i, seg in enumerate(segments):
            if check_cancel_callback and i % 10 == 0:
                check_cancel_callback()
            key = self._cache_key(src, tgt, seg.get('text', ''), cache_model,
                                  self._cache_settings(segments, i, optimize,
                                                       concise, context_applied))
            keys[i] = key
            cached = self._cache_get(key)
            if cached:
                s = seg.copy()
                s['translated_text'] = cached
                s['source_text'] = seg.get('text', '')
                s['translation'] = self._meta(plan, concise, "cached", None,
                                             context_applied=False)
                results[i] = s
            else:
                pending_idx.append(i)

        if pending_idx:
            logger.info("Translating %d/%d items (cache hit: %d).",
                        len(pending_idx), len(segments), len(segments) - len(pending_idx))
            # Context-aware translation is a single call: splitting it would cut
            # the dialogue context the backend is given. Everything else is
            # independent per line, so it runs in blocks and each block is
            # persisted before the next begins. A crash at line 800 of 1200 then
            # keeps the 800 already translated.
            step = len(pending_idx) if context_applied else TRANSLATION_FLUSH_EVERY
            for start in range(0, len(pending_idx), step):
                block = pending_idx[start:start + step]
                pending = [segments[i] for i in block]
                try:
                    hop_results = self._execute(plan, pending, hf_token, optimize,
                                                check_cancel_callback, concise)
                except BaseException:
                    # Persist the blocks already finished before the failure
                    # propagates; the current block is simply not recorded.
                    self._cache_flush()
                    raise

                for local_i, (idx, seg) in enumerate(zip(block, pending)):
                    res = hop_results[local_i]
                    s = seg.copy()
                    s['source_text'] = seg.get('text', '')
                    s['translated_text'] = res.text
                    status = "ok" if res.ok else "failed"
                    s['translation'] = self._meta(plan, concise, status, res.error,
                                                 context_applied=context_applied)
                    results[idx] = s

                    if res.ok and res.text:
                        self._cache_put(keys[idx], res.text)
                    else:
                        self.last_failures.append({
                            "index": idx,
                            "speaker_id": seg.get('speaker_id'),
                            "start": seg.get('start'),
                            "end": seg.get('end'),
                            "source_text": seg.get('text', ''),
                            "error": res.error,
                        })
                        logger.warning("Segment %d failed to translate (%s); left empty.",
                                       idx, res.error)

                # Durable checkpoint for this block.
                self._cache_flush()

        if self.last_failures:
            logger.warning("%d segment(s) failed translation; see last_failures for retry.",
                           len(self.last_failures))

        return results

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _execute(self, plan: TranslationPlan, segments: list[dict], hf_token,
                 optimize: bool, cancel, concise: str) -> list[SegmentResult]:
        """Run the plan, returning one SegmentResult per segment, in order."""
        texts = [s.get('text', '') for s in segments]
        speakers = [s.get('speaker_id') for s in segments]

        if plan.is_pivot:
            hop1 = self._run_hop(plan.hop_backends[0], texts, plan.steps[0], plan.steps[1],
                                 speakers, optimize, hf_token, cancel)
            # Feed only successful hop-1 output onward; a failed line stays failed.
            mid_texts = [r.text if r.ok else "" for r in hop1]
            hop2 = self._run_hop(plan.hop_backends[1], mid_texts, plan.steps[1], plan.steps[2],
                                 speakers, optimize, hf_token, cancel)
            out = []
            for a, b in zip(hop1, hop2):
                if not a.ok:
                    out.append(SegmentResult(text="", ok=False, error=f"pivot hop1: {a.error}",
                                             backend=plan.hop_backends[0]))
                else:
                    out.append(SegmentResult(text=b.text, ok=b.ok, error=b.error,
                                             backend=plan.hop_backends[1]))
            return out

        return self._run_hop(plan.backend, texts, plan.steps[0], plan.steps[1],
                             speakers, optimize, hf_token, cancel)

    def _run_hop(self, backend: str, texts: list[str], src_code: str, tgt_code: str,
                 speakers: list, optimize: bool, hf_token, cancel) -> list[SegmentResult]:
        if backend == "nllb":
            return self._run_nllb(texts, src_code, tgt_code, hf_token)
        if backend == "google":
            return self._run_google(texts, tgt_code)
        return self._run_llm(backend, texts, src_code, tgt_code, speakers,
                             optimize, hf_token, cancel)

    def _run_nllb(self, texts, src_code, tgt_code, hf_token) -> list[SegmentResult]:
        if not self.nllb_backend:
            return [SegmentResult("", False, "nllb backend not configured", "nllb") for _ in texts]
        src_flores = self.languages.source_flores(src_code)
        tgt_flores = self.languages.target_flores(tgt_code, for_speech=True)
        return self._batched(
            lambda chunk: self.nllb_backend.translate_batch(
                chunk, src_flores, tgt_flores, token=hf_token),
            texts, "nllb",
        )

    def _run_google(self, texts, tgt_code) -> list[SegmentResult]:
        translator, _ = self.get_google_translator(tgt_code)
        return self._batched(lambda chunk: [translator.translate(t) for t in chunk],
                             texts, "google")

    def _run_llm(self, model, texts, src_code, tgt_code, speakers, optimize,
                 hf_token, cancel) -> list[SegmentResult]:
        model_id_map = {
            "hymt": "tencent/HY-MT1.5-1.8B",
            "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "alma": "haoranxu/ALMA-7B-R",
        }
        mid = model_id_map.get(model, "tencent/HY-MT1.5-1.8B")

        if self.llm_translator and self.llm_translator.model_id != mid:
            self.llm_translator.unload_model()
            self.llm_translator = None
        if not self.llm_translator:
            self.llm_translator = LLMTranslator(mid)

        out: list[SegmentResult] = []
        if optimize:
            # Sequential, context-aware: neighbouring lines + speaker identity.
            for i, text in enumerate(texts):
                if cancel:
                    cancel()
                prev_text = out[i - 1].text if i > 0 and out[i - 1].ok else None
                next_text = texts[i + 1] if i + 1 < len(texts) else None
                try:
                    trans = self.llm_translator.translate_context_aware(
                        text, src_code, tgt_code,
                        prev_text=prev_text, next_text=next_text,
                        speaker_id=speakers[i] if i < len(speakers) else None,
                        token=hf_token,
                    )
                    out.append(self._check(text, trans, model))
                except Exception as exc:
                    out.append(SegmentResult("", False, str(exc), model))
        else:
            chunk_size = 4 if model in ("llama", "alma") else 8
            out = self._batched(
                lambda chunk: self.llm_translator.translate_batch(
                    chunk, src_code, tgt_code, token=hf_token),
                texts, model, chunk_size=chunk_size,
            )

        # Release VRAM so the TTS model can load.
        self.llm_translator.unload_model()
        return out

    # ------------------------------------------------------------------
    # Batch plumbing + per-segment failure isolation
    # ------------------------------------------------------------------
    def _batched(self, fn, texts, backend, chunk_size=8) -> list[SegmentResult]:
        """
        Run `fn(chunk) -> list[str]` over `texts` in chunks.

        A whole-chunk failure is retried one segment at a time so a single bad
        line cannot take out its neighbours. Segments that still fail are
        returned as `ok=False` with the error preserved, never as fake text.
        """
        out: list[SegmentResult] = []
        for start in range(0, len(texts), chunk_size):
            chunk = texts[start:start + chunk_size]
            try:
                translated = fn(chunk)
            except Exception as exc:
                logger.warning("%s chunk failed (%s); retrying per segment.", backend, exc)
                for text in chunk:
                    out.append(self._single(fn, text, backend))
                continue
            if translated is None or len(translated) != len(chunk):
                logger.warning("%s returned %s results for %d inputs; retrying per segment.",
                               backend, None if translated is None else len(translated), len(chunk))
                for text in chunk:
                    out.append(self._single(fn, text, backend))
                continue
            for text, trans in zip(chunk, translated):
                out.append(self._check(text, trans, backend))
        return out

    def _single(self, fn, text, backend) -> SegmentResult:
        try:
            translated = fn([text])
            if translated:
                return self._check(text, translated[0], backend)
            return SegmentResult("", False, "backend returned no output", backend)
        except Exception as exc:
            return SegmentResult("", False, str(exc), backend)

    @staticmethod
    def _check(source: str, translated, backend) -> SegmentResult:
        """Validate one backend output. Empty results are failures, not blanks."""
        if translated is None or not str(translated).strip():
            return SegmentResult("", False, "empty translation", backend)
        return SegmentResult(str(translated).strip(), True, None, backend)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    @staticmethod
    def _cache_key(src, tgt, text, model, settings=None):
        return translation_key(text, src, tgt, model, settings)

    def _cache_settings(self, segments, i, optimize, concise, context_applied):
        """
        The non-text inputs that change a line's translation.

        `optimize` and `concise` are always recorded. When the backend actually
        uses dialogue context, the neighbouring source lines and the speaker are
        recorded too, so editing a neighbour's text invalidates this line's
        cached translation rather than reusing one built from different context.
        Source-side context is used (not the previous translation) so the key is
        computable before anything is translated; the effect is conservative -
        it can miss a reuse, never wrongly reuse.
        """
        settings = {"optimize": bool(optimize), "concise": concise}
        if context_applied:
            seg = segments[i]
            settings["prev"] = segments[i - 1].get("text", "") if i > 0 else None
            settings["next"] = (segments[i + 1].get("text", "")
                                if i + 1 < len(segments) else None)
            settings["speaker"] = seg.get("speaker_id")
        return settings

    def _cache_get(self, key):
        """Read a cached translation from whichever store is authoritative."""
        if self.job_cache is not None:
            return self.job_cache.get_translation(key)
        return self.cache.get(key)

    def _cache_put(self, key, text):
        """Record a successful translation. Empty results are never cached."""
        if not text:
            return
        if self.job_cache is not None:
            self.job_cache.put_translation(key, text)
        else:
            self.cache[key] = text

    def _cache_flush(self):
        """Persist accumulated translations to the authoritative store."""
        if self.job_cache is not None:
            self.job_cache.flush_translations()
        else:
            self._save_cache()

    @staticmethod
    def _meta(plan: TranslationPlan, concise: str, status: str, error,
              context_applied: bool = False) -> dict:
        meta = plan.to_dict()
        meta.update({
            "concise": concise,
            "status": status,
            "error": error,
            "context_applied": context_applied,
        })
        return meta


if __name__ == "__main__":
    tr = Translator()
    # print(tr.translate_text("Hello world", "Spanish", model="hymt"))
