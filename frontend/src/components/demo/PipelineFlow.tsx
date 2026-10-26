/**
 * A concise, honest picture of what the pipeline does.
 *
 * The stages are the real ones the project runs. Diarization and lip-sync are
 * drawn as *optional* and explicitly labelled, because they only run when the
 * configuration asks for them and the models are installed — implying every run
 * uses them would overstate the product. Voice cloning is never implied: the
 * standard demo uses a generated voice.
 *
 * Purely presentational; it reads no backend state.
 */

interface FlowStage {
  label: string;
  detail?: string;
  optional?: boolean;
}

const STAGES: FlowStage[] = [
  { label: "Video", detail: "Source file" },
  { label: "Speech extraction", detail: "Audio pulled from the video" },
  { label: "Transcription", detail: "Speech recognized to text" },
  {
    label: "Speaker detection",
    detail: "Tells speakers apart so each keeps a voice",
    optional: true,
  },
  { label: "Translation", detail: "Text translated into the target language" },
  {
    label: "Speech generation",
    detail: "Target-language speech, speaker-aware when available",
  },
  { label: "Timing synchronization", detail: "Speech fitted to the original timing" },
  { label: "Background reconstruction", detail: "Music and effects kept from the source" },
  {
    label: "Lip-sync",
    detail: "Mouth movement matched to the new audio",
    optional: true,
  },
  { label: "Final dubbed video" },
];

export function PipelineFlow() {
  return (
    <ol className="pipeline-flow">
      {STAGES.map((stage) => (
        <li className="pipeline-flow-item" key={stage.label}>
          <div className="pipeline-flow-node">
            <span className="pipeline-flow-label">
              {stage.label}
              {stage.optional && (
                <span className="pipeline-flow-optional">optional</span>
              )}
            </span>
            {stage.detail && (
              <span className="pipeline-flow-detail">{stage.detail}</span>
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}
