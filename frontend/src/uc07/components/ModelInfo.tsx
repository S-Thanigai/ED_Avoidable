import { useEffect, useState } from "react";
import { getModelInfo, UC07ApiError } from "../api";
import type { ModelInfoResponse } from "../types";
import "./ModelInfo.css";

/** Technical/details panel backed by GET /model-info -- kept out of the
 * main decision flow, expandable on demand. */
export function ModelInfo() {
  const [open, setOpen] = useState(false);
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open || info || loading) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    getModelInfo()
      .then((result) => {
        if (!cancelled) setInfo(result);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof UC07ApiError || err instanceof Error ? err.message : "Could not load model info.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, info, loading]);

  return (
    <div className="model-info">
      <button
        type="button"
        className="model-info__toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        {open ? "Hide model details" : "About this model"}
      </button>
      {open && (
        <div className="model-info__panel">
          {loading && <p className="model-info__status">Loading model information…</p>}
          {error && <p className="model-info__status model-info__status--error">{error}</p>}
          {info && (
            <dl className="model-info__grid">
              <div>
                <dt>Model version</dt>
                <dd>{info.model_version}</dd>
              </div>
              <div>
                <dt>Algorithm</dt>
                <dd>{info.algorithm ?? "—"}</dd>
              </div>
              <div>
                <dt>Dataset</dt>
                <dd>{info.dataset_id}</dd>
              </div>
              <div>
                <dt>Synthetic data</dt>
                <dd>{info.synthetic_model ? "Yes" : "No"}</dd>
              </div>
              <div>
                <dt>Prediction horizon</dt>
                <dd>{info.prediction_horizon_days != null ? `${info.prediction_horizon_days} days` : "—"}</dd>
              </div>
              <div>
                <dt>Observation window</dt>
                <dd>{info.observation_window_days != null ? `${info.observation_window_days} days` : "—"}</dd>
              </div>
              <div>
                <dt>Feature count</dt>
                <dd>{info.feature_count}</dd>
              </div>
              <div>
                <dt>Intended use</dt>
                <dd>{info.intended_use ?? "—"}</dd>
              </div>
              {info.target_definition && (
                <div className="model-info__full-row">
                  <dt>Target</dt>
                  <dd>{info.target_definition}</dd>
                </div>
              )}
              {info.disclaimer && (
                <div className="model-info__full-row">
                  <dt>Disclaimer</dt>
                  <dd>{info.disclaimer}</dd>
                </div>
              )}
            </dl>
          )}
        </div>
      )}
    </div>
  );
}
