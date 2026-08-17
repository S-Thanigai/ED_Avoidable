import React from 'react';
import { AlertCircle, AlertTriangle, CheckCircle2 } from 'lucide-react';

export default function RiskBadge({ category, score, showScore = true, size = 'md' }) {
  const cat = String(category || 'Low').trim();
  
  let bgClass = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
  let Icon = CheckCircle2;

  if (cat === 'High') {
    bgClass = 'bg-rose-500/15 text-rose-400 border-rose-500/30';
    Icon = AlertCircle;
  } else if (cat === 'Medium') {
    bgClass = 'bg-amber-500/15 text-amber-400 border-amber-500/30';
    Icon = AlertTriangle;
  }

  const sizeClasses = size === 'sm' 
    ? 'px-2 py-0.5 text-xs gap-1' 
    : size === 'lg' 
    ? 'px-3.5 py-1.5 text-base gap-2 font-medium' 
    : 'px-2.5 py-1 text-xs font-medium gap-1.5';

  return (
    <span className={`inline-flex items-center rounded-full border ${bgClass} ${sizeClasses}`}>
      <Icon className={size === 'sm' ? 'w-3 h-3' : size === 'lg' ? 'w-5 h-5' : 'w-3.5 h-3.5'} />
      <span>{cat} Risk</span>
      {showScore && typeof score === 'number' && (
        <span className="opacity-80 border-l border-current/20 pl-1.5 font-mono">
          {score.toFixed(1)}
        </span>
      )}
    </span>
  );
}
