import torch
import numpy as np
import cv2
import os
import logging
import subprocess
import gc
from tqdm import tqdm
from pathlib import Path
import face_alignment

from src.processing.wav2lip_core.models import Wav2Lip
from src.processing.wav2lip_core import audio as audio_utils
from src.utils import config

logger = logging.getLogger(__name__)

# Constants for face detection and lip sync
FACE_DETECTION_MAX_DIM = 640  # Maximum dimension for face detection (speeds up CPU detection)
FACE_CROP_SCALE = 1.6  # Expansion factor for face bounding box in Wav2Lip
MEL_STEP_SIZE = 16  # Number of mel spectrogram frames per inference window

class Wav2LipSyncer:
    def __init__(self):
        self.model = None
        self.detector = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = 96
        self.batch_size = 32 # Default batch size
        self.model_path = Path("models/wav2lip/wav2lip_gan.pth")
        self.fallback_active = False
        self.sync_offset = 0 # Offset in mel steps (1 step ~ 12.5ms)
        self.restorer = None # GFPGAN Restorer

    def load_gfpgan(self):
        try:
            from gfpgan import GFPGANer
            # Model path
            # Model path
            gfpgan_path = Path("models/gfpgan/GFPGANv1.4.pth")
            
            model_arg = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
            if gfpgan_path.exists():
                model_arg = str(gfpgan_path)
            
            # Initialize GFPGAN
            # upscale=1 because we just want restoration, not full image upscaling (we do resizing later)
            self.restorer = GFPGANer(model_path=model_arg, 
                                     upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
            logger.info(f"GFPGAN Face Restorer Loaded (Source: {'Local' if gfpgan_path.exists() else 'URL'}).")
        except Exception as e:
            logger.warning(f"Failed to load GFPGAN: {e}")
            self.restorer = None
        except ImportError as e:
            logger.warning(f"Failed to import GFPGAN dependencies: {e}")
            self.restorer = None

    def load_model(self):
        if self.model is not None:
            return
            
        self.fallback_active = False # Reset on new load

        logger.info(f"Loading Wav2Lip model from {self.model_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Wav2Lip model not found at {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model = Wav2Lip()
        
        # Handle state dict mismatch if any (usually 'module.' prefix)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
            
        self.model.load_state_dict(new_s)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Note: Face detector is now lazy-loaded in _ensure_detector_loaded()
                                                     
    def _ensure_detector_loaded(self):
        """Lazy-load face detector with VRAM protection and CPU fallback."""
        if self.detector is not None:
            return
        
        # Explicitly clean memory before loading a large model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        detect_device = str(self.device).split(":")[0]
        logger.info(f"Loading Face Detection model (face_alignment) on {detect_device}...")
        
        try:
            self.detector = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                flip_input=False,
                device=detect_device
            )
        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                logger.warning(f"CUDA OOM during face detector init: {e}")
                logger.warning("Falling back to CPU for face detection...")
                self.fallback_active = True
                self.device = torch.device('cpu') # Update global device? Maybe just for detector?
                # Usually better to keep self.device as preferred, but if OOM, maybe we should switch?
                # For now, let's just make the detector CPU.
                self.detector = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D,
                    flip_input=False,
                    device='cpu'
                )
            else:
                raise

    def unload_model(self):
        """Unload models to free VRAM."""
        if self.model:
            del self.model
            self.model = None
        if self.detector:
            del self.detector
            self.detector = None
        if self.restorer:
            del self.restorer
            self.restorer = None
            
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Wav2Lip models unloaded.")

    def get_smooth_box(self, boxes, window_size=config.WAV2LIP_BOX_SMOOTH_WINDOW):
        """
        Smooths face bounding boxes over time using a moving average window.
        boxes: List of [x1, y1, x2, y2] or None
        """
        if not boxes:
            return boxes
            
        # 1. Fill None (Forward Fill)
        filled_boxes = []
        last_valid = None
        for box in boxes:
            if box is not None:
                last_valid = box
            filled_boxes.append(last_valid)
            
        # 2. Backward fill for start
        first_valid = None
        for box in filled_boxes:
            if box is not None:
                first_valid = box
                break
        
        if first_valid is None: 
            return boxes # All None - handled by caller
            
        final_filled = []
        for box in filled_boxes:
            if box is None:
                final_filled.append(first_valid)
            else:
                final_filled.append(box)
                
        # 3. Apply Smoothing
        smoothed_boxes = []
        half_window = window_size // 2
        
        current_boxes_np = np.array(final_filled) # (N, 4)
        
        for i in range(len(final_filled)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(final_filled), i + half_window + 1)
            
            window = current_boxes_np[start_idx:end_idx]
            avg_box = np.mean(window, axis=0).astype(int)
            smoothed_boxes.append(avg_box.tolist())
            
        return smoothed_boxes

    def _fill_missing_boxes(self, face_boxes):
        """Detailed fill logic that ensures every frame has a box if at least one was found."""
        # Find at least one valid box to start with
        curr_box = None
        for box in face_boxes:
            if box is not None:
                curr_box = box
                break
                
        if curr_box is None:
            return None # Process handled by caller
             
        # Forward Fill
        # We also need to handle the case where the start is None (Backward fill concept)
        # But since we initialize 'curr_box' with the first valid one found, 
        # any None at the start will get this value.
        
        filled = []
        for i in range(len(face_boxes)):
            if face_boxes[i] is None:
                filled.append(curr_box)
            else:
                curr_box = face_boxes[i]
                filled.append(curr_box)
                
        return filled

    def detect_faces(self, frames):
        """
        Detect faces in a list of frames.
        Returns a list of boxes [x1, y1, x2, y2] or None for each frame.
        Required by unit tests for fallback verification.
        """
        results = []
        for frame in tqdm(frames, desc="Face Detection (Manual)"):
            results.append(self._detect_single_frame(frame))
        return results

    def _detect_all_boxes(self, rgb_frame, scale_factor):
        """
        Every detected face in `rgb_frame` as a clamped `[x1, y1, x2, y2]` box.

        Returns a list, possibly empty. Boxes are expanded by `FACE_CROP_SCALE`,
        the Wav2Lip crop convention.
        """
        preds = self.detector.get_landmarks(rgb_frame)
        boxes = []
        for lm in (preds or []):
            if scale_factor != 1.0:
                lm = lm / scale_factor
            x_min, y_min = np.min(lm, axis=0)
            x_max, y_max = np.max(lm, axis=0)
            w_box = x_max - x_min
            h_box = y_max - y_min
            scale = FACE_CROP_SCALE
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            nw = w_box * scale
            nh = h_box * scale
            x1 = max(0, int(cx - nw / 2))
            y1 = max(0, int(cy - nh / 2))
            x2 = int(cx + nw / 2)
            y2 = int(cy + nh / 2)
            boxes.append([x1, y1, x2, y2])
        return boxes

    @staticmethod
    def _smooth_face_counts(counts, window_size=5):
        """
        Stabilise a per-frame face count for the ambiguity decision.

        A median filter removes a one-frame stray detection, then a max filter
        re-expands the result so a genuine multi-face frame at the edge of a
        shot is not averaged back down to one. Both directions matter: marking
        a single-actor frame ambiguous only skips one frame of lip-sync, while
        missing a multi-actor frame moves a guessed actor's mouth.
        """
        if not counts:
            return []
        half = max(0, window_size // 2)
        med = []
        for i in range(len(counts)):
            lo = max(0, i - half)
            hi = min(len(counts), i + half + 1)
            med.append(int(np.median(counts[lo:hi])))
        out = []
        for i in range(len(med)):
            lo = max(0, i - half)
            hi = min(len(med), i + half + 1)
            out.append(int(max(med[lo:hi])))
        return out

    def _detect_frame_boxes(self, frame):
        """
        Detect faces in `frame`.

        Returns `(all_boxes, chosen_box)`. `chosen_box` is the largest face when
        the frame has exactly one, otherwise None. `all_boxes` is always the
        full list, so the caller can record how crowded the frame was.
        """
        self._ensure_detector_loaded()
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb_frame.shape[:2]
            scale_factor = 1.0
            if max(h, w) > FACE_DETECTION_MAX_DIM:
                scale_factor = FACE_DETECTION_MAX_DIM / float(max(h, w))
                rgb_frame = cv2.resize(rgb_frame, (int(w * scale_factor), int(h * scale_factor)))

            boxes = self._detect_all_boxes(rgb_frame, scale_factor)
            if not boxes:
                return [], None
            boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            if len(boxes) != 1:
                return boxes, None
            x1, y1, x2, y2 = boxes[0]
            return boxes, [max(0, x1), max(0, y1),
                           min(frame.shape[1], x2), min(frame.shape[0], y2)]
        except Exception as e:
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                if not self.fallback_active:
                    logger.warning(f"CUDA Error during face detection: {e}. Switching to CPU fallback...")
                    self.device = torch.device("cpu")
                    self.fallback_active = True
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                    if hasattr(self, 'detector') and self.detector:
                        del self.detector
                        self.detector = None
                    self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                                                 flip_input=False, device='cpu')
                logger.info("Retrying frame on CPU...")
                return self._detect_frame_boxes(frame)
            logger.warning(f"Face detection error: {e}")
            return [], None

    def _detect_single_frame(self, frame):
        """
        Detect face in a single frame. Returns [x1, y1, x2, y2] or None.
        Handles OOM with CPU fallback for the specific frame.
        """
        self._ensure_detector_loaded()

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb_frame.shape[:2]
            scale_factor = 1.0
            if max(h, w) > FACE_DETECTION_MAX_DIM:
                scale_factor = FACE_DETECTION_MAX_DIM / float(max(h, w))
                rgb_frame = cv2.resize(rgb_frame, (int(w * scale_factor), int(h * scale_factor)))

            boxes = self._detect_all_boxes(rgb_frame, scale_factor)
            if not boxes:
                return None
            # Largest face first, then clamped to the frame.
            boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            x1, y1, x2, y2 = boxes[0]
            return [max(0, x1), max(0, y1),
                    min(frame.shape[1], x2), min(frame.shape[0], y2)]

        except Exception as e:
            # check for CUDA error
            if "CUDA" in str(e) or "out of memory" in str(e).lower():
                if not self.fallback_active:
                     logger.warning(f"CUDA Error during face detection: {e}. Switching to CPU fallback...")
                     self.device = torch.device("cpu")
                     self.fallback_active = True
                     os.environ["CUDA_VISIBLE_DEVICES"] = ""
                     if hasattr(self, 'detector') and self.detector:
                         del self.detector
                         self.detector = None
                     self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                                      flip_input=False, device='cpu')
                logger.info("Retrying frame on CPU...")
                return self._detect_single_frame(frame)
            else:
                logger.warning(f"Face detection error: {e}")
                return None


    def _load_frame_range(self, video_path: str | Path, start, end, pad_before=0, pad_after=0):
        """
        Load specific frame range from video file.
        Loads [start-pad_before, end+pad_after) frames.
        Returns the frames list and the index of 'start' within that list.
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        safe_start = max(0, start - pad_before)
        safe_end = min(total_frames, end + pad_after)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, safe_start)
        
        frames = []
        for _ in range(safe_end - safe_start):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        # Calculate where 'start' index is in this list
        relative_start_idx = start - safe_start
        
        return frames, relative_start_idx

    def _prepare_chunk_batches(self, chunk_frames, chunk_boxes, mel, fps, global_chunk_start, relative_start_idx, chunk_target_len):
        """
        Prepare inference batches for a frame chunk.
        
        chunk_frames: Frames loaded (including padding)
        chunk_boxes: Boxes for the TARGET frames (length = chunk_target_len)
        global_chunk_start: The global frame index where this chunk starts (the target start)
        relative_start_idx: The index in chunk_frames where global_chunk_start is
        chunk_target_len: Number of actual frames we want to process
        """
        input_batches = []
        mel_batches = []
        coords_batches = []
        
        current_faces = []
        current_mels = []
        current_coords = []
        
        mel_step_size = MEL_STEP_SIZE
        mel_idx_multiplier = 80.0 / fps
        
        # Iterate only over the target frames (ignore padding frames for the main loop)
        for i in range(chunk_target_len):
            global_frame_idx = global_chunk_start + i
            local_frame_idx = relative_start_idx + i
            
            # Audio window
            start_idx = int(global_frame_idx * mel_idx_multiplier) + self.sync_offset
            end_idx = start_idx + mel_step_size
            
            # Pad Mel if needed
            if end_idx > mel.shape[0]:
                 diff = end_idx - mel.shape[0]
                 m = np.concatenate((mel, np.zeros((diff, 80))), axis=0)
                 m_chunk = m[start_idx:end_idx]
            else:
                m_chunk = mel[start_idx:end_idx]
            
            m_chunk = np.transpose(m_chunk, (1, 0)) # (80, 16)
            
            # Video Window (5 frames: i-2 to i+2)
            window_frames = []
            audio_window = []
            
            # Helper to get frame from chunk_frames (handling boundary logic via clamping)
            def get_frame(rel_idx):
                # Clamp to available loaded frames
                # If padding wasn't enough (start of video), clamp to 0
                idx = max(0, min(len(chunk_frames)-1, rel_idx))
                return chunk_frames[idx]
            
            for j in range(-2, 3): # -2, -1, 0, 1, 2
                # Frame
                window_frames.append(get_frame(local_frame_idx + j))
                
                # Audio for this specific frame context
                ctx_global_idx = global_frame_idx + j
                if ctx_global_idx < 0: ctx_global_idx = 0 # Clamp audio too?
                
                s_idx = int(ctx_global_idx * mel_idx_multiplier)
                e_idx = s_idx + mel_step_size
                
                if e_idx > mel.shape[0]:
                    diff = e_idx - mel.shape[0]
                    m = np.concatenate((mel, np.zeros((diff, 80))), axis=0)
                    chunk = m[s_idx:e_idx]
                else:
                    chunk = mel[s_idx:e_idx]
                
                if chunk.size == 0: # Handle edge cases
                     chunk = np.zeros((80, 16))
                     
                chunk = np.transpose(chunk, (1, 0))
                audio_window.append(chunk)

            # Box. A `None` box (ambiguous shot, or no face) still gets a
            # placeholder crop so this frame's slot is kept and the results
            # stay aligned with the target frames; PASS 2 discards it.
            box = chunk_boxes[i]
            if box is None:
                placeholder = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                placeholder = np.concatenate((placeholder, placeholder), axis=2)
                window_np = np.transpose(np.stack([placeholder] * 5, axis=0), (0, 3, 1, 2))
                current_faces.append(window_np)
                current_mels.append(np.stack([np.zeros((80, 16))] * 5, axis=0))
                current_coords.append(None)
                if len(current_faces) >= self.batch_size:
                    input_batches.append(np.array(current_faces))
                    mel_batches.append(np.array(current_mels))
                    coords_batches.append(current_coords)
                    current_faces, current_mels, current_coords = [], [], []
                continue
            x1, y1, x2, y2 = box
            
            face_crops = []
            for wf in window_frames:
                f_crop = wf[y1:y2, x1:x2]
                f_crop = cv2.resize(f_crop, (self.img_size, self.img_size))
                face_crops.append(f_crop)
                
            processed_window = []
            for fc in face_crops:
                masked = fc.copy()
                masked[self.img_size//2:, :] = 0
                combined = np.concatenate((masked, fc), axis=2) 
                processed_window.append(combined)

            window_np = np.stack(processed_window, axis=0) # (5, 96, 96, 6)
            window_np = np.transpose(window_np, (0, 3, 1, 2)) # (5, 6, 96, 96)
            
            audio_np = np.stack(audio_window, axis=0) # (5, 80, 16)
            
            current_faces.append(window_np)
            current_mels.append(audio_np)
            current_coords.append((x1, y1, x2, y2))
            
            if len(current_faces) >= self.batch_size:
                input_batches.append(np.array(current_faces))
                mel_batches.append(np.array(current_mels))
                coords_batches.append(current_coords)
                current_faces = []
                current_mels = []
                current_coords = []
                
        if current_faces:
             input_batches.append(np.array(current_faces))
             mel_batches.append(np.array(current_mels))
             coords_batches.append(current_coords)
             
        return input_batches, mel_batches, coords_batches

    def _run_inference(self, input_batches, mel_batches, coords_batches):
        gen_frames = []
        gen_coords = []
        
        if not input_batches:
            return gen_frames, gen_coords

        try:
            with torch.no_grad():
                for faces, mels, coords in zip(input_batches, mel_batches, coords_batches):
                    # faces: (B, 5, 6, 96, 96)
                    img_batch = torch.FloatTensor(faces).to(self.device).float() / 255.0
                    mel_batch = torch.FloatTensor(mels).to(self.device).float().unsqueeze(2)

                    pred = self.model(mel_batch, img_batch)
                    pred = pred.cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.

                    for b_i in range(len(coords)):
                        p_frame = pred[b_i, 2] # Middle frame
                        gen_frames.append(p_frame.astype(np.uint8))
                        
                    gen_coords.extend(coords)
                    
        except RuntimeError as e:
            if "CUDA" in str(e) and self.device.type == "cuda":
                logger.warning(f"CUDA Error encountered in inference: {e}")
                logger.warning("Switching to CPU fallback for Wav2Lip...")
                
                # Force switch
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                torch.cuda.empty_cache()
                
                # Retry on CPU
                gen_frames = []
                gen_coords = [] # Clear partial results
                
                with torch.no_grad():
                    for faces, mels, coords in zip(input_batches, mel_batches, coords_batches):
                        img_batch = torch.FloatTensor(faces).to(self.device).float() / 255.0
                        mel_batch = torch.FloatTensor(mels).to(self.device).float().unsqueeze(2)
                        
                        pred = self.model(mel_batch, img_batch)
                        pred = pred.numpy().transpose(0, 2, 3, 4, 1) * 255.
                        
                        for b_i in range(len(coords)):
                            p_frame = pred[b_i, 2]
                            gen_frames.append(p_frame.astype(np.uint8))
                            
                        gen_coords.extend(coords)
            else:
                raise e
                
        return gen_frames, gen_coords

    def _blend_face(self, frame, g_crop, box, enhance_face):
        x1, y1, x2, y2 = box
        w_box = x2 - x1
        h_box = y2 - y1
        
        # Enhancement (GFPGAN)
        if enhance_face and self.restorer is not None:
            try:
                # Use paste_back=False to get the cropped face back
                _, restored_faces, _ = self.restorer.enhance(
                    g_crop, 
                    has_aligned=True, 
                    only_center_face=True, 
                    paste_back=False
                )
                if restored_faces:
                    g_crop = restored_faces[0]
            except Exception as e:
                logger.warning(f"GFPGAN enhancement failed: {e}")

        # Resize crop to fit the box
        try:
            g_crop_resized = cv2.resize(g_crop, (w_box, h_box), interpolation=cv2.INTER_LANCZOS4)
        except Exception:
             # If resize fails (zero sise?), just return original
             return frame
            
        # Seamless Clone
        center = (x1 + w_box//2, y1 + h_box//2)
        
        # Create a soft mask for fallback
        mask_fallback = np.zeros((h_box, w_box), dtype=np.float32)
        cv2.ellipse(mask_fallback, (w_box//2, h_box//2), (w_box//2 - 5, h_box//2 - 5), 0, 0, 360, 1.0, -1)
        mask_fallback = cv2.GaussianBlur(mask_fallback, (21, 21), 0)
        mask_fallback = mask_fallback[..., np.newaxis]

        # Check if the box touches the frame boundary
        # If it touches closely, seamlessClone can "bleed" artifacts from the edge
        margin = 5
        is_edge_contact = (x1 <= margin) or (y1 <= margin) or (x2 >= frame.shape[1] - margin) or (y2 >= frame.shape[0] - margin)
        
        use_seamless = True
        if is_edge_contact:
            # High risk of bleeding artifacts at edges
            use_seamless = False
            
        if use_seamless:
            try:
                # Standard seamless clone with a full mask usually works best for internal faces
                # But sometimes a slight erosion helps
                mask_sc = 255 * np.ones(g_crop_resized.shape, g_crop_resized.dtype)
                frame = cv2.seamlessClone(g_crop_resized, frame, mask_sc, center, cv2.NORMAL_CLONE)
            except Exception:
                # If seamless clone fails, fall back
                use_seamless = False
        
        if not use_seamless:
            try:
                # Alpha blending fallback (Manual "paste")
                roi = frame[y1:y2, x1:x2].astype(np.float32)
                fg = g_crop_resized.astype(np.float32)
                
                blended = (fg * mask_fallback + roi * (1.0 - mask_fallback)).astype(np.uint8)
                frame[y1:y2, x1:x2] = blended
            except Exception:
                 # Last resort: hard paste (rarely looks good but better than crash)
                 try:
                    frame[y1:y2, x1:x2] = g_crop_resized
                 except:
                    pass
                
        return frame

    def sync_lips(self, video_path: str | Path, audio_path: str | Path, output_path: str | Path, enhance_face: bool = False) -> str:
        """Streaming lip sync that processes video in chunks."""
        if enhance_face and self.restorer is None:
            self.load_gfpgan()
        self.load_model()
        
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        
        # --- PASS 1: Stream face detection (O(1) memory per frame) ---
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Pass 1: Detecting faces in {total_frames} frames (streaming)...")
        raw_face_boxes = []
        # Faces per frame, so PASS 2 can tell a single-actor shot from a
        # multi-actor shot and leave the ambiguous ones alone.
        raw_face_counts = []

        for frame_idx in tqdm(range(total_frames), desc="Face Detection"):
            ret, frame = cap.read()
            if not ret:
                raw_face_boxes.append(None)
                raw_face_counts.append(0)
                continue

            boxes, box = self._detect_frame_boxes(frame)
            raw_face_boxes.append(box)
            raw_face_counts.append(len(boxes))
            del frame  # Release immediately


        cap.release()
        gc.collect()
        
        # Apply smoothing
        face_boxes = self.get_smooth_box(raw_face_boxes, window_size=5)
        face_boxes = self._fill_missing_boxes(face_boxes)

        # `_fill_missing_boxes` treats an ambiguous frame (two or more faces) as
        # a gap and fills it from its neighbours, which would silently pick an
        # actor. Re-mark those frames so PASS 2 leaves them alone. Counts are
        # smoothed with a short window so one stray detection does not punch a
        # hole in an otherwise single-actor shot.
        ambiguous = self._smooth_face_counts(raw_face_counts, window_size=5)
        if face_boxes is not None:
            for idx, count in enumerate(ambiguous):
                if count >= 2:
                    face_boxes[idx] = None
        
        if face_boxes is None:
            logger.error("No faces detected in ANY frame. Skipping Wav2Lip processing (returning original).")
            # Fallback join
            subprocess.run([
                "ffmpeg", "-y", "-v", "warning",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy", "-c:a", "aac",
                output_path
            ], check=True)
            return output_path
        
        # Process Audio (Mel)
        mel = audio_utils.wav2mel(str(audio_path))
        
        # --- PASS 2: Chunked inference ---
        CHUNK_SIZE = 300  # ~10 seconds at 30fps
        
        temp_video_path = str(config.TEMP_DIR / "wav2lip_chunked.mp4")
        out_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        logger.info(f"Pass 2: Processing {total_frames} frames in chunks of {CHUNK_SIZE}...")
        
        # We need to process every frame using the detected boxes
        for chunk_start in tqdm(range(0, total_frames, CHUNK_SIZE), desc="Processing Chunks"):
            chunk_end = min(chunk_start + CHUNK_SIZE, total_frames)
            chunk_len = chunk_end - chunk_start
            
            # Load chunk frames with padding for temporal context (Wav2Lip inputs 5 frames)
            # We need padded frames for inference input, but we only generate for chunk_len
            frames, rel_start = self._load_frame_range(video_path, chunk_start, chunk_end, pad_before=2, pad_after=2)
            
            chunk_boxes = face_boxes[chunk_start:chunk_end]
            
            # Prepare batches for this chunk
            input_batches, mel_batches, coords_batches = self._prepare_chunk_batches(
                frames, chunk_boxes, mel, fps, 
                global_chunk_start=chunk_start, 
                relative_start_idx=rel_start,
                chunk_target_len=chunk_len
            )
            
            # Run inference
            gen_frames, gen_coords = self._run_inference(input_batches, mel_batches, coords_batches)
            
            # Blend and write output
            # We must be careful: if inference failed/dropped frames, zip stops.
            # Ideally gen_frames count matches chunk_len.
            
            # The generated frames correspond exactly to the chunk_boxes (and thus target chunk frames)
            # because we iterated chunk_target_len in _prepare_chunk_batches
            
            for i, (g_crop, box) in enumerate(zip(gen_frames, gen_coords)):
                # Map back to the loaded frames list using rel_start
                # The i-th result corresponds to frames[rel_start + i]
                frame = frames[rel_start + i]
                if box is None:
                    # Ambiguous shot (or a gap): no reliable speaker-to-face
                    # association, so the frame is passed through untouched
                    # rather than having a guessed actor's mouth moved.
                    out_writer.write(frame)
                    continue
                final_frame = self._blend_face(frame, g_crop, box, enhance_face)
                out_writer.write(final_frame)
            
            # Aggressive cleanup
            del frames, input_batches, mel_batches, gen_frames, gen_coords
            gc.collect()
        
        out_writer.release()
        
        # --- PASS 3: Merge audio ---
        logger.info("Pass 3: Merging audio...")
        subprocess.run([
            "ffmpeg", "-y", "-v", "warning",
            "-i", temp_video_path,
            "-i", str(audio_path),
            "-c:v", "copy", "-c:a", "aac",
            output_path
        ], check=True)
        
        # Cleanup temp
        if os.path.exists(temp_video_path):
             os.remove(temp_video_path)
             
        return output_path
