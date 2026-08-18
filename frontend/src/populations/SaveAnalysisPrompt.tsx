import { useState } from "react";
import type { UploadFiles } from "../types";
import { PopulationsApiError, saveAnalysis } from "./api";
import type { PopulationSummary } from "./types";
import "./SaveAnalysisPrompt.css";

type SaveState = "idle" | "saving" | "saved" | "failed";

/** Shown after a successful CSV analysis (Uc07View), below the existing
 * results -- never in place of them. Resubmits the SAME files + the
 * SAME index_date the on-screen analysis already used (never "today"
 * again), so what gets saved is guaranteed to match what's on screen --
 * see backend/uc07_pipeline.py, which both /uc07/decide and
 * /populations/save-analysis call through. Saving is entirely optional:
 * "Not Now" does nothing and the current analysis stays exactly as it
 * was. */
export function SaveAnalysisPrompt({
  files,
  indexDate,
  safetyContextFile,
  memberCount,
}: {
  files: UploadFiles;
  indexDate: string;
  safetyContextFile: File | null;
  memberCount: number;
}) {
  const [dismissed, setDismissed] = useState(false);
  const [name, setName] = useState("");
  const [state, setState] = useState<SaveState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<PopulationSummary | null>(null);

  if (dismissed) return null;

  const handleSave = async () => {
    const trimmed = name.trim();
    if (!trimmed) {
      setError("Give this population a name before saving.");
      return;
    }
    setState("saving");
    setError(null);
    try {
      const population = await saveAnalysis({ name: trimmed, files, indexDate, safetyContextFile });
      setSaved(population);
      setState("saved");
    } catch (err) {
      setError(err instanceof PopulationsApiError || err instanceof Error ? err.message : "Save failed.");
      setState("failed");
    }
  };

  if (state === "saved" && saved) {
    return (
      <div className="save-analysis-prompt save-analysis-prompt--success" role="status">
        <span className="save-analysis-prompt__icon" aria-hidden="true">
          ✓
        </span>
        <div>
          <strong>Saved successfully.</strong>{" "}
          <span>
            "{saved.name}" ({saved.member_count.toLocaleString()} members) is now in your Saved Populations.
          </span>
        </div>
        <button type="button" className="save-analysis-prompt__dismiss" onClick={() => setDismissed(true)}>
          Dismiss
        </button>
      </div>
    );
  }

  return (
    <div className="save-analysis-prompt" aria-label="Save this analysis">
      <div className="save-analysis-prompt__header">
        <h3>Save this analysis?</h3>
        <p>Save this analyzed population ({memberCount.toLocaleString()} members) to your account so you can return to it without uploading the files again.</p>
      </div>

      <label className="save-analysis-prompt__field">
        <span>Population name</span>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. August ED Population"
          disabled={state === "saving"}
          maxLength={200}
        />
      </label>

      {error && (
        <div className="save-analysis-prompt__error" role="alert">
          {error}
        </div>
      )}

      <div className="save-analysis-prompt__actions">
        <button type="button" className="save-analysis-prompt__secondary" onClick={() => setDismissed(true)} disabled={state === "saving"}>
          Not Now
        </button>
        <button type="button" className="save-analysis-prompt__primary" onClick={handleSave} disabled={state === "saving"}>
          {state === "saving" ? "Saving…" : "Save to Database"}
        </button>
      </div>
    </div>
  );
}
