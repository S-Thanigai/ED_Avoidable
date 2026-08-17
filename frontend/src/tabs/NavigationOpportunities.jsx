import React, { useState, useMemo } from 'react';
import PatientTable from '../components/PatientTable';
import { Compass, UserCheck, Car, Video, Building2, Filter } from 'lucide-react';
import { getNavigationOpportunityBreakdown } from '../utils/aggregations';

export default function NavigationOpportunities({ patients = [], onSelectPatient }) {
  const [selectedTypeFilter, setSelectedTypeFilter] = useState('ALL');

  // Filter for patients with non-empty navigation opportunities AND safety_guardrail_flag === false
  const eligiblePatients = useMemo(() => {
    return patients.filter(
      p => Array.isArray(p.navigation_opportunities) &&
           p.navigation_opportunities.length > 0 &&
           !p.safety_guardrail_flag
    );
  }, [patients]);

  const opportunityBreakdown = useMemo(() => {
    return getNavigationOpportunityBreakdown(patients);
  }, [patients]);

  // Filter by active opportunity type selection
  const filteredPatients = useMemo(() => {
    if (selectedTypeFilter === 'ALL') return eligiblePatients;

    return eligiblePatients.filter(p => {
      const oppsStr = p.navigation_opportunities.map(o => String(o).toLowerCase()).join(' ');
      if (selectedTypeFilter === 'PCP') return oppsStr.includes('pcp');
      if (selectedTypeFilter === 'Transportation') return oppsStr.includes('transport');
      if (selectedTypeFilter === 'Telehealth') return oppsStr.includes('telehealth');
      if (selectedTypeFilter === 'UrgentCare') return oppsStr.includes('urgent');
      return true;
    });
  }, [eligiblePatients, selectedTypeFilter]);

  const typeConfig = [
    { key: 'ALL', name: 'All Opportunities', icon: Compass, count: eligiblePatients.length },
    { key: 'PCP', name: 'PCP Navigation', icon: UserCheck, count: opportunityBreakdown.find(b => b.key === 'PCP')?.count || 0 },
    { key: 'Transportation', name: 'Transportation Support', icon: Car, count: opportunityBreakdown.find(b => b.key === 'Transportation')?.count || 0 },
    { key: 'Telehealth', name: 'Telehealth Access', icon: Video, count: opportunityBreakdown.find(b => b.key === 'Telehealth')?.count || 0 },
    { key: 'UrgentCare', name: 'Urgent Care Navigation', icon: Building2, count: opportunityBreakdown.find(b => b.key === 'UrgentCare')?.count || 0 },
  ];

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header Banner */}
      <div className="glass-panel p-6 rounded-2xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-purple-500/30 bg-purple-950/10">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Compass className="w-5 h-5 text-purple-400" />
            <h2 className="text-lg font-bold text-slate-100">
              Proactive Care Navigation Opportunities
            </h2>
          </div>
          <p className="text-xs text-slate-300 max-w-2xl leading-relaxed">
            High-risk patients clear of high-severity emergency guardrail flags who qualify for outpatient navigation, telehealth, transportation, or urgent care outreach.
          </p>
        </div>

        <div className="px-4 py-2 rounded-xl bg-slate-900/80 border border-slate-700 text-center">
          <div className="text-xs text-slate-400 font-medium">Eligible Patients</div>
          <div className="text-xl font-bold font-mono text-purple-300">{eligiblePatients.length}</div>
        </div>
      </div>

      {/* Opportunity Type Filter Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        {typeConfig.map((item) => {
          const Icon = item.icon;
          const isSelected = selectedTypeFilter === item.key;

          return (
            <button
              key={item.key}
              onClick={() => setSelectedTypeFilter(item.key)}
              className={`p-3.5 rounded-xl border text-left flex flex-col justify-between transition-all duration-200 ${
                isSelected
                  ? 'bg-purple-500/15 border-purple-500/50 text-purple-200 shadow-lg shadow-purple-500/10'
                  : 'glass-card border-slate-800 hover:border-slate-700 text-slate-400 hover:text-slate-200'
              }`}
            >
              <div className="flex items-center justify-between">
                <Icon className={`w-4 h-4 ${isSelected ? 'text-purple-400' : 'text-slate-400'}`} />
                <span className={`text-xs font-mono font-bold px-1.5 py-0.5 rounded ${isSelected ? 'bg-purple-500/20 text-purple-300' : 'bg-slate-900 text-slate-400'}`}>
                  {item.count}
                </span>
              </div>
              <div className="mt-2 text-xs font-semibold tracking-tight">
                {item.name}
              </div>
            </button>
          );
        })}
      </div>

      {/* Patient Table */}
      <PatientTable
        patients={filteredPatients}
        onSelectPatient={onSelectPatient}
        pageSize={10}
        emptyMessage="No patients found for this navigation opportunity filter."
      />
    </div>
  );
}
