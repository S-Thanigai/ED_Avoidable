import React from 'react';
import { ShieldAlert, Info } from 'lucide-react';

export default function SafetyBanner({ message, compact = false }) {
  const defaultMessage =
    "Clinical/Emergency care indicators present — DO NOT discourage ED. This patient's risk score reflects utilization pattern only, not a recommendation to avoid the ED.";

  const displayMessage = message || defaultMessage;

  if (compact) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/15 border border-amber-500/30 text-amber-300 text-xs font-medium">
        <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0" />
        <span>Safety Guardrail Triggered</span>
      </div>
    );
  }

  return (
    <div className="relative overflow-hidden rounded-xl border border-amber-500/40 bg-gradient-to-r from-amber-950/40 via-amber-900/30 to-slate-900/80 p-4 sm:p-5 shadow-lg shadow-amber-950/20 backdrop-blur-md">
      <div className="flex items-start gap-3.5">
        <div className="p-2.5 rounded-xl bg-amber-500/20 text-amber-400 border border-amber-500/40 shrink-0 mt-0.5">
          <ShieldAlert className="w-6 h-6 animate-pulse" />
        </div>
        
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-semibold text-amber-200 uppercase tracking-wider">
              Clinical Safety Guardrail Active
            </h4>
            <span className="px-2 py-0.5 text-[10px] uppercase tracking-wider font-bold bg-amber-500/20 text-amber-300 rounded border border-amber-500/40">
              High Acuity Indicators Found
            </span>
          </div>

          <p className="text-sm text-amber-100/90 leading-relaxed font-medium">
            "{displayMessage}"
          </p>

          <div className="pt-2 text-xs text-amber-300/70 flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5" />
            <span>Ethical Care Principle: Emergency severity overrides algorithm risk recommendations. Never delay urgent emergency evaluation.</span>
          </div>
        </div>
      </div>
    </div>
  );
}
