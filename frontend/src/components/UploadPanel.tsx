import { useRef, useState, type DragEvent } from "react";
import type { UploadFiles } from "../types";
import "./UploadPanel.css";

interface DropzoneConfig {
  key: keyof UploadFiles;
  label: string;
  hint: string;
}

const ZONES: DropzoneConfig[] = [
  { key: "members", label: "Members CSV", hint: "member_id, age, gender, access barriers…" },
  { key: "edVisits", label: "ED Visits CSV", hint: "visit_date, diagnosis, admitted, red_flag…" },
  { key: "care", label: "Care History CSV", hint: "visit_date, care_type (PCP/UC/Telehealth)…" },
];

interface UploadPanelProps {
  files: UploadFiles;
  onFileChange: (key: keyof UploadFiles, file: File | null) => void;
  onRun: () => void;
  loading: boolean;
}

function Dropzone({
  config,
  file,
  onFileChange,
  disabled,
}: {
  config: DropzoneConfig;
  file: File | null;
  onFileChange: (file: File | null) => void;
  disabled: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const accept = (candidate: File | undefined) => {
    if (!candidate) return;
    if (!candidate.name.toLowerCase().endsWith(".csv")) return;
    onFileChange(candidate);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragOver(false);
    if (disabled) return;
    accept(event.dataTransfer.files?.[0]);
  };

  return (
    <div
      className={`dropzone${dragOver ? " dropzone--over" : ""}${file ? " dropzone--filled" : ""}`}
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      role="button"
      tabIndex={0}
      aria-disabled={disabled}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        hidden
        disabled={disabled}
        onChange={(e) => accept(e.target.files?.[0])}
      />
      <span className="dropzone__label">{config.label}</span>
      {file ? (
        <span className="dropzone__filename">{file.name}</span>
      ) : (
        <>
          <span className="dropzone__cta">Drop CSV or click to browse</span>
          <span className="dropzone__hint">{config.hint}</span>
        </>
      )}
      {file && (
        <button
          type="button"
          className="dropzone__clear"
          onClick={(e) => {
            e.stopPropagation();
            onFileChange(null);
          }}
          aria-label={`Remove ${config.label}`}
        >
          ×
        </button>
      )}
    </div>
  );
}

export function UploadPanel({ files, onFileChange, onRun, loading }: UploadPanelProps) {
  const allSelected = files.members && files.edVisits && files.care;

  return (
    <section className="upload-panel">
      <div className="upload-panel__grid">
        {ZONES.map((zone) => (
          <Dropzone
            key={zone.key}
            config={zone}
            file={files[zone.key]}
            disabled={loading}
            onFileChange={(file) => onFileChange(zone.key, file)}
          />
        ))}
      </div>
      <div className="upload-panel__actions">
        <button
          type="button"
          className="run-button"
          disabled={!allSelected || loading}
          onClick={onRun}
        >
          {loading ? (
            <>
              <span className="spinner" aria-hidden="true" /> Running analysis…
            </>
          ) : (
            "Run risk analysis"
          )}
        </button>
        <span className="upload-panel__note">
          Files stay on this request only — nothing is stored server-side beyond the
          scoring run.
        </span>
      </div>
    </section>
  );
}
