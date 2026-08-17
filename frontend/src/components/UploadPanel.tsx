import { useRef, useState, type DragEvent, type KeyboardEvent, type ReactNode } from "react";
import type { UploadFiles } from "../types";
import "./UploadPanel.css";

interface DropzoneConfig {
  key: keyof UploadFiles;
  label: string;
  hint: string;
  icon: "members" | "ed" | "care";
}

const ZONES: DropzoneConfig[] = [
  { key: "members", label: "Members CSV", hint: "member_id, age, gender, access barriers…", icon: "members" },
  { key: "edVisits", label: "ED Visits CSV", hint: "visit_date, diagnosis, admitted, red_flag…", icon: "ed" },
  { key: "care", label: "Care History CSV", hint: "visit_date, care_type (PCP/UC/Telehealth)…", icon: "care" },
];

/** Small, dependency-free inline icon set -- deliberately not pulling in
 * an icon library for three glyphs. Purely decorative (aria-hidden);
 * the text label always carries the actual meaning. */
function ZoneIcon({ kind }: { kind: DropzoneConfig["icon"] }) {
  const paths: Record<DropzoneConfig["icon"], ReactNode> = {
    members: (
      <>
        <circle cx="12" cy="8" r="3.2" />
        <path d="M5 20c0-3.5 3-6 7-6s7 2.5 7 6" />
      </>
    ),
    ed: (
      <>
        <path d="M12 3v6M9 6h6" />
        <rect x="4" y="9" width="16" height="12" rx="2" />
      </>
    ),
    care: (
      <>
        <path d="M20 8.5c0-2.5-2-4.5-4.5-4.5-1.4 0-2.7.7-3.5 1.8C11.2 4.7 9.9 4 8.5 4 6 4 4 6 4 8.5c0 5 8 10.5 8 10.5s8-5.5 8-10.5Z" />
      </>
    ),
  };
  return (
    <svg
      className="dropzone__icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[kind]}
    </svg>
  );
}

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
  const [error, setError] = useState<string | null>(null);

  const accept = (candidate: File | undefined) => {
    if (!candidate) return;
    if (!candidate.name.toLowerCase().endsWith(".csv")) {
      setError("Only .csv files are accepted.");
      return;
    }
    setError(null);
    onFileChange(candidate);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragOver(false);
    if (disabled) return;
    accept(event.dataTransfer.files?.[0]);
  };

  const openPicker = () => {
    if (!disabled) inputRef.current?.click();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openPicker();
    }
  };

  return (
    <div
      className={`dropzone dropzone--${config.icon}${dragOver ? " dropzone--over" : ""}${file ? " dropzone--filled" : ""}${error ? " dropzone--error" : ""}`}
      onClick={openPicker}
      onKeyDown={handleKeyDown}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      aria-label={file ? `${config.label}: ${file.name} selected. Activate to replace.` : `${config.label}: no file selected. Activate to browse or drop a CSV.`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        hidden
        disabled={disabled}
        onChange={(e) => accept(e.target.files?.[0])}
      />

      <div className="dropzone__top">
        <span className="dropzone__icon-chip" aria-hidden="true">
          <ZoneIcon kind={config.icon} />
        </span>
        <span className="dropzone__label">{config.label}</span>
        {file && !error && (
          <span className="dropzone__success" aria-hidden="true" title="File selected">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3.5 8.5 6.5 11.5 12.5 4.5" />
            </svg>
          </span>
        )}
      </div>

      {file ? (
        <span className="dropzone__filename">{file.name}</span>
      ) : (
        <>
          <span className="dropzone__cta">Drag &amp; drop, or click to browse</span>
          <span className="dropzone__hint">{config.hint}</span>
        </>
      )}

      {error && (
        <span className="dropzone__error-text" role="alert">
          {error}
        </span>
      )}

      {file && (
        <button
          type="button"
          className="dropzone__clear"
          onClick={(e) => {
            e.stopPropagation();
            setError(null);
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
    <section className="upload-panel" aria-label="Historical data inputs">
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
          className={`run-button${allSelected ? " run-button--ready" : ""}`}
          disabled={!allSelected || loading}
          onClick={onRun}
        >
          {loading ? (
            <>
              <span className="spinner" aria-hidden="true" /> Running analysis…
            </>
          ) : (
            "Run Risk Analysis"
          )}
        </button>
        <span className="upload-panel__note">
          Files stay on this request only — nothing is stored server-side beyond the scoring run.
        </span>
      </div>
    </section>
  );
}
