import React from 'react';

export default function KpiCard({ title, value, subtitle, icon: Icon, color = 'sky', trend }) {
  const colorStyles = {
    sky: 'from-sky-500/10 to-sky-500/5 text-sky-400 border-sky-500/20',
    rose: 'from-rose-500/10 to-rose-500/5 text-rose-400 border-rose-500/20',
    amber: 'from-amber-500/10 to-amber-500/5 text-amber-400 border-amber-500/20',
    emerald: 'from-emerald-500/10 to-emerald-500/5 text-emerald-400 border-emerald-500/20',
    purple: 'from-purple-500/10 to-purple-500/5 text-purple-400 border-purple-500/20',
  };

  const currentStyle = colorStyles[color] || colorStyles.sky;

  return (
    <div className="glass-panel glass-panel-hover p-5 rounded-xl flex flex-col justify-between relative overflow-hidden group">
      {/* Decorative subtle background gradient blur */}
      <div className={`absolute -right-6 -top-6 w-24 h-24 bg-gradient-to-br ${currentStyle} rounded-full blur-2xl opacity-40 group-hover:opacity-70 transition-opacity`} />

      <div className="flex items-center justify-between z-10">
        <span className="text-xs font-medium text-slate-400 tracking-wider uppercase">{title}</span>
        {Icon && (
          <div className={`p-2 rounded-lg bg-slate-900/60 border border-slate-700/50 ${currentStyle}`}>
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>

      <div className="mt-4 z-10">
        <div className="text-3xl font-bold text-slate-100 tracking-tight font-mono">
          {value}
        </div>
        {subtitle && (
          <div className="mt-1 text-xs text-slate-400 flex items-center gap-1.5">
            {subtitle}
          </div>
        )}
      </div>
    </div>
  );
}
