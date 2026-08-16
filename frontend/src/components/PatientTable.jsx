import React, { useState, useMemo } from 'react';
import { Search, ChevronLeft, ChevronRight, ShieldAlert, ArrowUpDown, ExternalLink } from 'lucide-react';
import RiskBadge from './RiskBadge';
import { formatFeatureName } from './ShapFactorChart';

export default function PatientTable({
  patients = [],
  onSelectPatient,
  pageSize = 10,
  emptyMessage = 'No matching patients found.',
}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortField, setSortField] = useState('risk_score');
  const [sortDirection, setSortDirection] = useState('desc');
  const [currentPage, setCurrentPage] = useState(1);

  // Search & Filter
  const filteredPatients = useMemo(() => {
    if (!searchTerm.trim()) return patients;
    const term = searchTerm.toLowerCase();
    return patients.filter(
      p => String(p.member_id).toLowerCase().includes(term) ||
           String(p.risk_category).toLowerCase().includes(term)
    );
  }, [patients, searchTerm]);

  // Sort
  const sortedPatients = useMemo(() => {
    return [...filteredPatients].sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];

      if (typeof valA === 'string') valA = valA.toLowerCase();
      if (typeof valB === 'string') valB = valB.toLowerCase();

      if (valA < valB) return sortDirection === 'asc' ? -1 : 1;
      if (valA > valB) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });
  }, [filteredPatients, sortField, sortDirection]);

  // Pagination (10 rows per page per user selection)
  const totalPages = Math.max(1, Math.ceil(sortedPatients.length / pageSize));
  const paginatedPatients = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return sortedPatients.slice(start, start + pageSize);
  }, [sortedPatients, currentPage, pageSize]);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(prev => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  return (
    <div className="space-y-4">
      {/* Search & Filter Header Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search by Member ID..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setCurrentPage(1);
            }}
            className="w-full pl-10 pr-4 py-2 bg-slate-900/80 border border-slate-700/60 rounded-xl text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-sky-500/60 focus:ring-2 focus:ring-sky-500/20 transition-all"
          />
        </div>

        <div className="text-xs text-slate-400 font-medium self-end sm:self-center">
          Showing <span className="text-slate-200 font-bold">{sortedPatients.length}</span> patients
        </div>
      </div>

      {/* Table */}
      <div className="glass-panel rounded-xl overflow-hidden border border-slate-800 shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-300">
            <thead className="bg-slate-900/80 text-xs text-slate-400 uppercase tracking-wider border-b border-slate-800">
              <tr>
                <th
                  onClick={() => handleSort('member_id')}
                  className="py-3.5 px-4 font-semibold cursor-pointer hover:text-slate-200 select-none"
                >
                  <div className="flex items-center gap-1.5">
                    Member ID
                    <ArrowUpDown className="w-3.5 h-3.5 opacity-60" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('risk_score')}
                  className="py-3.5 px-4 font-semibold cursor-pointer hover:text-slate-200 select-none"
                >
                  <div className="flex items-center gap-1.5">
                    Risk Score & Category
                    <ArrowUpDown className="w-3.5 h-3.5 opacity-60" />
                  </div>
                </th>
                <th className="py-3.5 px-4 font-semibold">Safety Guardrail</th>
                <th className="py-3.5 px-4 font-semibold">Top Positive Drivers</th>
                <th className="py-3.5 px-4 font-semibold">Care Navigation</th>
                <th className="py-3.5 px-4 text-right font-semibold">Action</th>
              </tr>
            </thead>

            <tbody className="divide-y divide-slate-800/60">
              {paginatedPatients.length > 0 ? (
                paginatedPatients.map((patient) => {
                  const topPos = Array.isArray(patient.top_positive_factors) ? patient.top_positive_factors : [];
                  const navOpps = Array.isArray(patient.navigation_opportunities) ? patient.navigation_opportunities : [];

                  return (
                    <tr
                      key={patient.member_id}
                      onClick={() => onSelectPatient && onSelectPatient(patient)}
                      className="hover:bg-slate-800/40 cursor-pointer transition-colors duration-150 group"
                    >
                      <td className="py-3.5 px-4 font-mono font-medium text-slate-200 flex items-center gap-2">
                        <span>{patient.member_id}</span>
                        {typeof patient.age === 'number' && (
                          <span className="text-xs font-sans text-slate-400 font-normal">
                            ({patient.age}y {patient.gender})
                          </span>
                        )}
                      </td>

                      <td className="py-3.5 px-4">
                        <RiskBadge
                          category={patient.risk_category}
                          score={patient.risk_score}
                          size="sm"
                        />
                      </td>

                      <td className="py-3.5 px-4">
                        {patient.safety_guardrail_flag ? (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-500/15 border border-amber-500/30 text-amber-300 text-xs font-medium">
                            <ShieldAlert className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                            Flagged
                          </span>
                        ) : (
                          <span className="text-xs text-slate-500 font-medium">Clear</span>
                        )}
                      </td>

                      <td className="py-3.5 px-4 max-w-xs">
                        {topPos.length > 0 ? (
                          <div className="text-xs text-slate-300 truncate">
                            <span className="text-slate-400 font-medium">
                              {formatFeatureName(topPos[0].feature)}
                            </span>
                            {topPos.length > 1 && (
                              <span className="text-slate-500 ml-1">+{topPos.length - 1} more</span>
                            )}
                          </div>
                        ) : (
                          <span className="text-xs text-slate-500">None</span>
                        )}
                      </td>

                      <td className="py-3.5 px-4">
                        {navOpps.length > 0 ? (
                          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-sky-500/10 text-sky-400 border border-sky-500/20">
                            {navOpps.length} Opportunities
                          </span>
                        ) : (
                          <span className="text-xs text-slate-500">None</span>
                        )}
                      </td>

                      <td className="py-3.5 px-4 text-right">
                        <span className="inline-flex items-center gap-1 text-xs text-sky-400 font-medium group-hover:text-sky-300 group-hover:underline">
                          View Detail
                          <ExternalLink className="w-3.5 h-3.5" />
                        </span>
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-slate-400 text-sm">
                    {emptyMessage}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Bar */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-4 py-3 bg-slate-900/60 border-t border-slate-800 text-xs text-slate-400">
            <div>
              Page <span className="font-semibold text-slate-200">{currentPage}</span> of{' '}
              <span className="font-semibold text-slate-200">{totalPages}</span>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
                className="p-1.5 rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-slate-200 transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                disabled={currentPage === totalPages}
                className="p-1.5 rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-slate-200 transition-colors"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
