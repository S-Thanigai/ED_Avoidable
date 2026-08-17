import React from 'react';
import { Download, BarChart3, TrendingUp, Compass, ShieldAlert } from 'lucide-react';
import {
  getRiskScoreHistogram,
  getRiskByChronicCount,
  getAccessBarrierImpact,
  getTrendsSummaryStats,
} from '../utils/aggregations';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from 'recharts';

export default function TrendsReports({ patients = [] }) {
  const histogramData = getRiskScoreHistogram(patients);
  const chronicRiskData = getRiskByChronicCount(patients);
  const barrierImpactData = getAccessBarrierImpact(patients);
  const stats = getTrendsSummaryStats(patients);

  const handleExportSummary = () => {
    if (!stats) return;

    const summaryPayload = {
      generatedAt: new Date().toISOString(),
      totalCohortCount: patients.length,
      ...stats,
      histogramDistribution: histogramData,
      riskByChronicConditions: chronicRiskData,
      accessBarrierImpact: barrierImpactData,
    };

    const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(
      JSON.stringify(summaryPayload, null, 2)
    )}`;
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', jsonString);
    downloadAnchor.setAttribute('download', `ed_risk_summary_report_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header & Export Action */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-extrabold text-slate-100 tracking-tight">
            Population Risk Trends & Clinical Reports
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Statistical distribution of ED utilization risk scores, chronic condition correlation, and social access barriers.
          </p>
        </div>

        <button
          onClick={handleExportSummary}
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold text-xs transition-colors shadow-lg shadow-sky-500/10 shrink-0"
        >
          <Download className="w-4 h-4" />
          <span>Export Summary Report (JSON)</span>
        </button>
      </div>

      {/* Summary Stats Grid */}
      {stats && (
        <div className="glass-panel p-6 rounded-2xl space-y-4">
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            Cohort Statistical Summary
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-4">
            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium">Mean Risk Score</div>
              <div className="text-xl font-bold font-mono text-sky-400 mt-1">{stats.meanRiskScore}</div>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium">Median Risk Score</div>
              <div className="text-xl font-bold font-mono text-sky-400 mt-1">{stats.medianRiskScore}</div>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium">Min / Max Score</div>
              <div className="text-sm font-bold font-mono text-slate-200 mt-1">
                {stats.minRiskScore} / {stats.maxRiskScore}
              </div>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium">High Risk Mean Age</div>
              <div className="text-xl font-bold font-mono text-rose-400 mt-1">{stats.meanHighRiskAge}y</div>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium">Low Risk Mean Age</div>
              <div className="text-xl font-bold font-mono text-emerald-400 mt-1">{stats.meanLowRiskAge}y</div>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium">High / Low Ratio</div>
              <div className="text-sm font-bold font-mono text-purple-300 mt-1">
                {stats.highRiskPatientCount} : {stats.lowRiskPatientCount}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Chart 1: Risk Score Distribution Histogram */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            Risk Score Histogram Distribution
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Binned patient count across 0-100 risk score ranges.
          </p>
        </div>

        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={histogramData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
              <XAxis dataKey="range" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="glass-card p-3 rounded-lg border border-slate-700 text-xs shadow-xl">
                        <div className="font-bold text-slate-200">Score Bucket: {label}</div>
                        <div className="text-sky-400 font-mono mt-1">
                          Patients: <span className="font-bold">{payload[0].value}</span>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="count" fill="#38bdf8" radius={[4, 4, 0, 0]}>
                {histogramData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.min >= 70 ? '#ef4444' : entry.min >= 40 ? '#f59e0b' : '#10b981'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 text-[11px] text-slate-400">
          <span className="font-semibold text-slate-300">Caption:</span> The bimodal score distribution highlights a clear separation between stable low-utilizers and high-acuity chronic patients.
        </div>
      </div>

      {/* Grid 2: Chronic Count Impact & Transportation Barrier Impact */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chronic Condition Count vs Risk Score */}
        <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
              Average Risk Score by Chronic Condition Count
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Impact of accumulating chronic conditions on average risk score.
            </p>
          </div>

          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chronicRiskData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
                <XAxis dataKey="label" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} domain={[0, 100]} />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="glass-card p-3 rounded-lg border border-slate-700 text-xs shadow-xl">
                          <div className="font-bold text-slate-200">{label}</div>
                          <div className="text-sky-400 font-mono mt-1">
                            Avg Risk Score: <span className="font-bold">{payload[0].value}</span>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="avgScore" fill="#a855f7" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 text-[11px] text-slate-400">
            <span className="font-semibold text-slate-300">Caption:</span> Patients with 3+ chronic conditions show exponentially higher mean ED utilization risk.
          </div>
        </div>

        {/* Transportation Barrier Impact */}
        <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
              Transportation Access Barrier Impact
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Comparing average risk score for patients with vs without transportation barriers.
            </p>
          </div>

          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barrierImpactData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.4} />
                <XAxis dataKey="group" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} domain={[0, 100]} />
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="glass-card p-3 rounded-lg border border-slate-700 text-xs shadow-xl">
                          <div className="font-bold text-slate-200">{label}</div>
                          <div className="text-amber-400 font-mono mt-1">
                            Avg Risk Score: <span className="font-bold">{payload[0].value}</span>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="avgScore" fill="#f43f5e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 text-[11px] text-slate-400">
            <span className="font-semibold text-slate-300">Caption:</span> Lack of transportation functions as a key friction point, driving reliance on ambulance and ED services.
          </div>
        </div>
      </div>
    </div>
  );
}
