import React from 'react';
import { Activity, AlertCircle, Compass, User, BarChart3 } from 'lucide-react';

export default function TabNav({ activeTab, onTabChange, highRiskCount = 0, navOppCount = 0, selectedPatient = null }) {
  const tabs = [
    { id: 'overview', label: 'Population Overview', icon: Activity },
    { id: 'high-risk', label: 'High-Risk Patients', icon: AlertCircle, badge: highRiskCount, badgeColor: 'bg-rose-500/20 text-rose-300 border-rose-500/30' },
    { id: 'navigation', label: 'Navigation Opportunities', icon: Compass, badge: navOppCount, badgeColor: 'bg-purple-500/20 text-purple-300 border-purple-500/30' },
    { id: 'detail', label: 'Patient Detail', icon: User, indicator: selectedPatient ? selectedPatient.member_id : null },
    { id: 'trends', label: 'Trends & Reports', icon: BarChart3 },
  ];

  return (
    <div className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <nav className="flex space-x-1 sm:space-x-2 overflow-x-auto py-2 scrollbar-none">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;

            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`flex items-center gap-2 px-3.5 py-2.5 rounded-xl text-xs sm:text-sm font-medium whitespace-nowrap transition-all duration-200 ${
                  isActive
                    ? 'bg-sky-500/15 text-sky-300 border border-sky-500/30 shadow-lg shadow-sky-500/10'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-sky-400' : 'text-slate-400'}`} />
                <span>{tab.label}</span>

                {typeof tab.badge === 'number' && tab.badge > 0 && (
                  <span className={`ml-1 px-2 py-0.5 text-[10px] font-bold font-mono rounded-full border ${tab.badgeColor}`}>
                    {tab.badge}
                  </span>
                )}

                {tab.indicator && (
                  <span className="ml-1 px-1.5 py-0.5 text-[10px] font-mono rounded bg-slate-800 text-sky-400 border border-slate-700">
                    {tab.indicator}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>
    </div>
  );
}
