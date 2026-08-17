import React, { useMemo } from 'react';
import PatientTable from '../components/PatientTable';
import { AlertCircle, ShieldAlert } from 'lucide-react';

export default function HighRiskPatients({ patients = [], onSelectPatient }) {
  // Filter for patients in the High Risk category
  const highRiskPatients = useMemo(() => {
    return patients.filter(p => p.risk_category === 'High');
  }, [patients]);

  const totalHigh = highRiskPatients.length;
  const safetyCount = highRiskPatients.filter(p => p.safety_guardrail_flag).length;

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header Info Banner */}
      <div className="glass-panel p-6 rounded-2xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-rose-500/30 bg-rose-950/10">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-rose-400" />
            <h2 className="text-lg font-bold text-slate-100">
              High-Risk ED Utilization Cohort
            </h2>
          </div>
          <p className="text-xs text-slate-300 max-w-2xl leading-relaxed">
            Patients prioritized with a risk score $\ge 70.0$. Click any patient row to open their clinical SHAP explanations, safety overrides, and care navigation recommendations.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="px-3.5 py-1.5 rounded-xl bg-slate-900/80 border border-slate-700 text-center">
            <div className="text-xs text-slate-400 font-medium">Cohort Count</div>
            <div className="text-lg font-bold font-mono text-rose-400">{totalHigh}</div>
          </div>
          <div className="px-3.5 py-1.5 rounded-xl bg-slate-900/80 border border-slate-700 text-center">
            <div className="text-xs text-slate-400 font-medium">Safety Overrides</div>
            <div className="text-lg font-bold font-mono text-amber-400">{safetyCount}</div>
          </div>
        </div>
      </div>

      {/* Patient Table Component */}
      <PatientTable
        patients={highRiskPatients}
        onSelectPatient={onSelectPatient}
        pageSize={10}
        emptyMessage="No High-Risk patients found matching current filters."
      />
    </div>
  );
}
