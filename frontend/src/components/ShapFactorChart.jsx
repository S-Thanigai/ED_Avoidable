import React from 'react';
import { ArrowUpRight, ArrowDownRight, Info } from 'lucide-react';

export function formatFeatureName(rawName = '') {
  if (!rawName) return '';
  const str = String(rawName);

  if (str.startsWith('diagnosis_')) {
    const dx = str.replace('diagnosis_', '').replace(/_/g, ' ');
    return `Diagnosis: ${dx}`;
  }

  const nameMap = {
    num_chronic_conditions: 'Chronic Condition Count',
    clinical_burden: 'Clinical Burden Index',
    access_burden: 'Access Barrier Friction Index',
    pcp_distance_miles: 'Distance to PCP (Miles)',
    urgent_care_distance_miles: 'Distance to Urgent Care (Miles)',
    transportation_barrier: 'Transportation Barrier Present',
    telehealth_available: 'Telehealth Option Available',
    ever_high_acuity_triage: 'History of High Acuity Triage (Level 1-2)',
    ever_admitted: 'History of Hospital Admission',
    ever_icu: 'History of ICU Stay',
    ever_major_procedure: 'History of Major Procedure',
    ever_red_flag: 'History of Red Flag Indicator',
    care_PCP: 'PCP Outpatient Visits',
    care_Telehealth: 'Telehealth Consultations',
    care_Urgent_Care: 'Urgent Care Visits',
    care_Care_Management: 'Care Management Encounters',
    days_since_last_care: 'Days Since Last Outpatient Care',
    alternative_care_visits: 'Alternative Care Visits',
    total_non_ED_care: 'Total Non-ED Care Interactions',
    age: 'Patient Age',
    gender_M: 'Male Gender Indicator',
    diabetes: 'Diabetes Flag',
    copd: 'COPD Flag',
    hypertension: 'Hypertension Flag',
    chf: 'Congestive Heart Failure Flag',
    asthma: 'Asthma Flag',
    ckd: 'Chronic Kidney Disease Flag',
  };

  return nameMap[str] || str.replace(/_/g, ' ');
}

export default function ShapFactorChart({ topPositive = [], topNegative = [] }) {
  const hasPos = Array.isArray(topPositive) && topPositive.length > 0;
  const hasNeg = Array.isArray(topNegative) && topNegative.length > 0;

  if (!hasPos && !hasNeg) {
    return (
      <div className="glass-card p-6 rounded-xl text-center text-slate-400 text-sm">
        No specific SHAP explainability factors available for this patient profile.
      </div>
    );
  }

  // Find max magnitude for relative scaling
  const maxMagnitude = Math.max(
    ...topPositive.map(f => Math.abs(f.shap_value || 0)),
    ...topNegative.map(f => Math.abs(f.shap_value || 0)),
    0.01
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            SHAP Risk Explainability Drivers
          </h4>
          <p className="text-xs text-slate-400 mt-0.5">
            Key clinical and demographic factors driving risk probability up or down.
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs font-medium">
          <div className="flex items-center gap-1.5 text-rose-400">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block" />
            <span>Increases Risk</span>
          </div>
          <div className="flex items-center gap-1.5 text-emerald-400">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block" />
            <span>Decreases Risk (Protective)</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Risk Increasing Factors (Positive SHAP) */}
        <div className="glass-card p-4 rounded-xl border-rose-500/20 bg-rose-950/10 space-y-3">
          <div className="flex items-center gap-2 text-rose-400 text-xs font-semibold uppercase tracking-wider pb-2 border-b border-rose-500/20">
            <ArrowUpRight className="w-4 h-4" />
            <span>Top Risk-Increasing Drivers (+SHAP)</span>
          </div>

          {hasPos ? (
            <div className="space-y-3 pt-1">
              {topPositive.map((factor, idx) => {
                const val = parseFloat(factor.shap_value) || 0;
                const widthPct = Math.min((Math.abs(val) / maxMagnitude) * 100, 100);
                return (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="font-medium text-slate-200">
                        {formatFeatureName(factor.feature)}
                      </span>
                      <span className="font-mono text-rose-400 font-semibold">
                        +{val.toFixed(3)}
                      </span>
                    </div>
                    <div className="w-full h-2 rounded-full bg-slate-900 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-rose-500 to-amber-500 transition-all duration-500"
                        style={{ width: `${widthPct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-xs text-slate-400 italic py-2">No positive risk-increasing drivers identified.</p>
          )}
        </div>

        {/* Risk Decreasing Factors (Negative SHAP) */}
        <div className="glass-card p-4 rounded-xl border-emerald-500/20 bg-emerald-950/10 space-y-3">
          <div className="flex items-center gap-2 text-emerald-400 text-xs font-semibold uppercase tracking-wider pb-2 border-b border-emerald-500/20">
            <ArrowDownRight className="w-4 h-4" />
            <span>Top Risk-Decreasing Drivers (-SHAP)</span>
          </div>

          {hasNeg ? (
            <div className="space-y-3 pt-1">
              {topNegative.map((factor, idx) => {
                const val = parseFloat(factor.shap_value) || 0;
                const widthPct = Math.min((Math.abs(val) / maxMagnitude) * 100, 100);
                return (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="font-medium text-slate-200">
                        {formatFeatureName(factor.feature)}
                      </span>
                      <span className="font-mono text-emerald-400 font-semibold">
                        {val.toFixed(3)}
                      </span>
                    </div>
                    <div className="w-full h-2 rounded-full bg-slate-900 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-500"
                        style={{ width: `${widthPct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-xs text-slate-400 italic py-2">No negative protective drivers identified.</p>
          )}
        </div>
      </div>
    </div>
  );
}
