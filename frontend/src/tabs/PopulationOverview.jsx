import React from 'react';
import { Users, AlertCircle, AlertTriangle, CheckCircle2, ShieldAlert } from 'lucide-react';
import KpiCard from '../components/KpiCard';
import {
  computePopulationKpis,
  getRiskDistributionChartData,
  getChronicPrevalenceStackedChartData,
  RISK_COLORS,
} from '../utils/aggregations';
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';

export default function PopulationOverview({ patients = [] }) {
  const kpis = computePopulationKpis(patients);
  const riskPieData = getRiskDistributionChartData(patients);
  const chronicStackedData = getChronicPrevalenceStackedChartData(patients);

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Page Title & Intro */}
      <div>
        <h2 className="text-xl font-extrabold text-slate-100 tracking-tight">
          Population Health Overview
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          High-level executive metrics on emergency department utilization risk and safety guardrails across {kpis.totalPatients} scored patients.
        </p>
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <KpiCard
          title="Total Scored Patients"
          value={kpis.totalPatients.toLocaleString()}
          subtitle="Full cohort dataset"
          icon={Users}
          color="sky"
        />
        <KpiCard
          title="High Risk (>=70)"
          value={kpis.highRiskCount.toLocaleString()}
          subtitle={`${kpis.highRiskPct}% of population`}
          icon={AlertCircle}
          color="rose"
        />
        <KpiCard
          title="Medium Risk (40-69.9)"
          value={kpis.medRiskCount.toLocaleString()}
          subtitle={`${kpis.medRiskPct}% of population`}
          icon={AlertTriangle}
          color="amber"
        />
        <KpiCard
          title="Low Risk (<40)"
          value={kpis.lowRiskCount.toLocaleString()}
          subtitle={`${kpis.lowRiskPct}% of population`}
          icon={CheckCircle2}
          color="emerald"
        />
        <KpiCard
          title="Safety Guardrail Active"
          value={kpis.safetyFlagCount.toLocaleString()}
          subtitle={`${kpis.safetyFlagPct}% severe indicators`}
          icon={ShieldAlert}
          color="purple"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Risk Category Donut Chart (5 cols) */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-2xl flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
                Risk Stratification Breakdown
              </h3>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-slate-800 text-sky-400 border border-slate-700">
                Donut Chart
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Distribution of predicted ED utilization risk categories.
            </p>
          </div>

          <div className="h-64 my-4">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskPieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={85}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {riskPieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} stroke="#0f172a" strokeWidth={2} />
                  ))}
                </Pie>
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="glass-card p-3 rounded-lg border border-slate-700 text-xs shadow-xl">
                          <div className="font-bold" style={{ color: data.color }}>{data.name}</div>
                          <div className="text-slate-200 mt-1">
                            Count: <span className="font-mono font-bold">{data.value}</span> ({data.pct}%)
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend
                  verticalAlign="bottom"
                  height={36}
                  formatter={(value, entry) => (
                    <span className="text-xs text-slate-300 font-medium">{value}</span>
                  )}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 text-[11px] text-slate-400 leading-relaxed">
            <span className="font-semibold text-slate-300">Caption:</span> High risk patients ($\ge 70$ score) represent the primary candidates for care navigation and outreach resources.
          </div>
        </div>

        {/* Chronic Burden vs Risk Categories Stacked Bar Chart (User Preference!) (7 cols) */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-2xl flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
                Chronic Burden vs Risk Categories
              </h3>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-slate-800 text-sky-400 border border-slate-700">
                Stacked Bar Chart
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Prevalence of chronic conditions segmented across High, Medium, and Low risk buckets.
            </p>
          </div>

          <div className="h-64 my-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chronicStackedData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
                <XAxis dataKey="condition" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="glass-card p-3 rounded-lg border border-slate-700 text-xs shadow-xl space-y-1">
                          <div className="font-bold text-slate-100">{label} Breakdown</div>
                          {payload.map((item, idx) => (
                            <div key={idx} className="flex justify-between gap-4" style={{ color: item.color }}>
                              <span>{item.name} Risk:</span>
                              <span className="font-mono font-bold">{item.value}</span>
                            </div>
                          ))}
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend verticalAlign="bottom" height={36} formatter={(value) => <span className="text-xs text-slate-300">{value} Risk</span>} />
                <Bar dataKey="High" stackId="a" fill={RISK_COLORS.High} />
                <Bar dataKey="Medium" stackId="a" fill={RISK_COLORS.Medium} />
                <Bar dataKey="Low" stackId="a" fill={RISK_COLORS.Low} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 text-[11px] text-slate-400 leading-relaxed">
            <span className="font-semibold text-slate-300">Caption:</span> Patients with chronic respiratory (COPD, Asthma) and cardiovascular (CHF, Hypertension) conditions exhibit a disproportionately higher share of High Risk scores.
          </div>
        </div>
      </div>
    </div>
  );
}
