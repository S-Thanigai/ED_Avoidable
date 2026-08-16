import { useEffect, useState } from "react";
import { explainMember } from "../api";
import type { PatientRow, ShapExplanation, UploadFiles } from "../types";
import { CarePill, RiskPill } from "./Pills";
import "./PatientDetailPanel.css";

const CONDITION_LABELS: Record<string, string> = {
  diabetes: "Diabetes",
  copd: "COPD",
  hypertension: "Hypertension",
  chf: "CHF",
  asthma: "Asthma",
  ckd: "CKD",
};

function ShapBar({ feature, value, max }: { feature: string; value: number; max: number }) {
  const increases = value > 0;
  return (
    <div className="shap-bar">
      <div className="shap-bar__meta">
        <span className="shap-bar__feature">{feature.replace(/_/g, " ")}</span>
        <span className={`shap-bar__value ${increases ? "shap-bar__value--up" : "shap-bar__value--down"}`}>
          {increases ? "+" : ""}
          {value.toFixed(3)}
        </span>
      </div>
      <div className="shap-bar__track">
        <div
          className={`shap-bar__fill ${increases ? "shap-bar__fill--up" : "shap-bar__fill--down"}`}
          style={{ width: `${Math.min(100, (Math.abs(value) / (max || 1)) * 100)}%` }}
        />
      </div>
    </div>
  );
}

export function PatientDetailPanel({
  row,
  files,
  onClose,
}: {
  row: PatientRow;
  files: UploadFiles;
  onClose: () => void;
}) {
  const [shap, setShap] = useState<ShapExplanation | null>(null);
  const [shapError, setShapError] = useState<string | null>(null);
  const [shapLoading, setShapLoading] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    let cancelled = false;
    setShap(null);
    setShapError(null);
    setShapLoading(true);

    explainMember(files, row.member_id)
      .then((result) => {
        if (!cancelled) setShap(result);
      })
      .catch((err) => {
        if (!cancelled) {
          setShapError(err instanceof Error ? err.message : "Could not load explanation.");
        }
      })
      .finally(() => {
        if (!cancelled) setShapLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [files, row.member_id]);

  const shapFeatures = shap
    ? [
        { feature: shap.shap_feature_1, value: shap.shap_value_1 },
        { feature: shap.shap_feature_2, value: shap.shap_value_2 },
        { feature: shap.shap_feature_3, value: shap.shap_value_3 },
      ].filter((f) => f.feature)
    : [];
  const maxAbs = Math.max(...shapFeatures.map((f) => Math.abs(f.value)), 0.0001);

  const conditions = Object.entries(CONDITION_LABELS).filter(
    ([key]) => Number((row as unknown as Record<string, number>)[key]) === 1,
  );

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()}>
        <div className="drawer__header">
          <div>
            <span className="drawer__eyebrow">Member</span>
            <h2 className="drawer__title">{row.member_id}</h2>
          </div>
          <button type="button" className="drawer__close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="drawer__body">
          <section className="drawer__section">
            <div className="drawer__risk-row">
              <RiskPill category={row.risk_category} />
              <span className="drawer__risk-score tabular">
                {row.risk_score.toFixed(1)} / 100 risk score
              </span>
            </div>
            <p className="drawer__risk-prob">
              Model probability of frequent avoidable-pattern ED use:{" "}
              <strong>{(row.risk_probability * 100).toFixed(1)}%</strong>
            </p>
          </section>

          <section className="drawer__section recommendation-card">
            <span className="drawer__section-title">Recommended next step</span>
            <CarePill care={row.recommended_alternative_care} />
            <p className="recommendation-card__reason">{row.alternative_care_reason}</p>
            <p className="recommendation-card__disclaimer">
              This is a navigation suggestion for care management outreach, not a
              clinical or emergency-necessity determination. It never overrides the
              member's own judgment to seek emergency care.
            </p>
          </section>

          <section className="drawer__section">
            <span className="drawer__section-title">Top risk drivers (SHAP)</span>
            {shapLoading && <p className="shap-status">Computing explanation…</p>}
            {shapError && !shapLoading && <p className="shap-status shap-status--error">{shapError}</p>}
            {!shapLoading && !shapError && shapFeatures.length > 0 && (
              <>
                <div className="shap-bars">
                  {shapFeatures.map((f) => (
                    <ShapBar key={f.feature} feature={f.feature} value={f.value} max={maxAbs} />
                  ))}
                </div>
                <div className="shap-legend">
                  <span>
                    <i className="shap-legend__dot shap-legend__dot--up" /> increases risk
                  </span>
                  <span>
                    <i className="shap-legend__dot shap-legend__dot--down" /> reduces risk
                  </span>
                </div>
              </>
            )}
          </section>

          <section className="drawer__section">
            <span className="drawer__section-title">Patient snapshot</span>
            <dl className="snapshot-grid">
              <div>
                <dt>Age / gender</dt>
                <dd>
                  {row.age} · {row.gender}
                </dd>
              </div>
              <div>
                <dt>Days since last ED visit</dt>
                <dd>{row.days_since_last_ED ?? "—"}</dd>
              </div>
              <div>
                <dt>Days since last non-ED care</dt>
                <dd>{row.days_since_last_care ?? "—"}</dd>
              </div>
              <div>
                <dt>Alternative-care visits (UC + Telehealth)</dt>
                <dd>{row.alternative_care_visits}</dd>
              </div>
              <div>
                <dt>Total non-ED care touchpoints</dt>
                <dd>{row.total_non_ED_care}</dd>
              </div>
              <div>
                <dt>Access burden</dt>
                <dd>{row.access_burden}</dd>
              </div>
              <div>
                <dt>Clinical burden</dt>
                <dd>{row.clinical_burden}</dd>
              </div>
              <div>
                <dt>Telehealth available</dt>
                <dd>{row.telehealth_available ? "Yes" : "No"}</dd>
              </div>
              <div>
                <dt>Transportation barrier</dt>
                <dd>{row.transportation_barrier ? "Yes" : "No"}</dd>
              </div>
              <div>
                <dt>PCP distance</dt>
                <dd>{row.pcp_distance_miles.toFixed(1)} mi</dd>
              </div>
              <div>
                <dt>Urgent care distance</dt>
                <dd>{row.urgent_care_distance_miles.toFixed(1)} mi</dd>
              </div>
            </dl>
          </section>

          {conditions.length > 0 && (
            <section className="drawer__section">
              <span className="drawer__section-title">Chronic conditions</span>
              <div className="condition-chips">
                {conditions.map(([key, label]) => (
                  <span className="condition-chip" key={key}>
                    {label}
                  </span>
                ))}
              </div>
            </section>
          )}
        </div>
      </aside>
    </div>
  );
}
