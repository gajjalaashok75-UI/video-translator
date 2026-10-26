/**
 * Source-video picker.
 *
 * Accepts a drag-and-drop or a click-to-browse, then hands the chosen `File` to
 * the caller, which owns the actual upload. The accepted extensions come from
 * the backend (`GET /api/system` exposes `ffmpeg_available`, but the extension
 * list is enforced server-side); this only mirrors the common video types so the
 * picker is useful, and the server remains the authority — a rejected file comes
 * back as an `ApiError` the caller shows.
 */

import { useRef, useState } from "react";
import { cx, formatBytes } from "../../utils/format";
import { IconUpload } from "../common/Icons";

const ACCEPT =
  ".mp4,.mkv,.mov,.avi,.webm,.m4v,.mpg,.mpeg,.wmv,video/*";

export function UploadDropzone({
  file,
  onFile,
  disabled,
}: {
  file: File | null;
  onFile: (file: File) => void;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function pick(candidate: File | undefined | null) {
    if (!candidate) return;
    onFile(candidate);
  }

  return (
    <div
      className={cx("dropzone", dragging && "dropzone-active", disabled && "dropzone-disabled")}
      onDragOver={(event) => {
        if (disabled) return;
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        if (disabled) return;
        event.preventDefault();
        setDragging(false);
        pick(event.dataTransfer.files?.[0]);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="visually-hidden"
        disabled={disabled}
        onChange={(event) => {
          pick(event.target.files?.[0]);
          // Reset so choosing the same file again still fires a change.
          event.target.value = "";
        }}
      />

      {file ? (
        <div className="dropzone-file">
          <IconUpload size={20} />
          <div className="dropzone-file-main">
            <span className="dropzone-file-name" title={file.name}>
              {file.name}
            </span>
            <span className="dropzone-file-meta">{formatBytes(file.size)}</span>
          </div>
          <button
            type="button"
            className="btn btn-ghost"
            disabled={disabled}
            onClick={() => inputRef.current?.click()}
          >
            Replace
          </button>
        </div>
      ) : (
        <button
          type="button"
          className="dropzone-prompt"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
        >
          <IconUpload size={24} />
          <span className="dropzone-title">Drop a video here</span>
          <span className="dropzone-hint">
            or click to browse — MP4, MKV, MOV, WebM
          </span>
        </button>
      )}
    </div>
  );
}
