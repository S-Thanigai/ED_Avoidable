import React, { useState, useEffect } from 'react';
import { Search, User, MapPin, Bus, PhoneCall, ShieldAlert, Sparkles, Check, X, Loader2 } from 'lucide-react';
import RiskBadge from '../components/RiskBadge';
import SafetyBanner from '../components/SafetyBanner';
import ShapFactorChart from '../components/ShapFactorChart';
import NavigationOpportunityCard from '../components/NavigationOpportunityCard';

export default function PatientDetail({ selectedPatient, patients = [], onSelectPatient, uploadedFiles }) {
  const [searchInput, setSearchInput] = useState('');
  
  // Dynamic SHAP state
  const [shapLoading, setShapLoading] = useState(false);
  const [shapError, setShapError] = useState(null);
  const [topPositive, setTopPositive] = useState([]);
  const [topNegative, setTopNegative] = useState([]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (!searchInput.trim()) return;
    const found = patients.find(
      p => String(p.member_id).toLowerCase() === searchInput.trim().toLowerCase()
    );
    if (found) {
      onSelectPatient(found);
    } else {
      alert(`Patient ID "${searchInput}" not found in loaded cohort.`);
    }
  };

  // On-demand SHAP reasoning loading
  useEffect(() => {
    if (!selectedPatient) return;

    // Check if SHAP data is already parsed and stored in the object
    const hasPositive = Array.isArray(selectedPatient.top_positive_factors) && selectedPatient.top_positive_factors.length > 0;
    const hasNegative = Array.isArray(selectedPatient.top_negative_factors) && selectedPatient.top_negative_factors.length > 0;

    if (hasPositive || hasNegative) {
      setTopPositive(selectedPatient.top_positive_factors || []);
      setTopNegative(selectedPatient.top_negative_factors || []);
      setShapLoading(false);
      setShapError(null);
      return;
    }

    // Try to fetch on-demand SHAP calculations
    let active = true;
    setShapLoading(true);
    setShapError(null);

    const fetchShapExplanation = async () => {
      try {
        const API_BASE_URL = window.location.origin.includes('517') 
          ? 'http://127.0.0.1:8001' 
          : window.location.origin;

        const formData = new FormData();
        formData.append('member_id', selectedPatient.member_id);

        if (uploadedFiles && uploadedFiles.members && uploadedFiles.edVisits && uploadedFiles.care) {
          formData.append('members_file', uploadedFiles.members);
          formData.append('ed_visits_file', uploadedFiles.edVisits);
          formData.append('care_file', uploadedFiles.care);
        }

        const response = await fetch(`${API_BASE_URL}/explain-member`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`AI explanation request failed (HTTP ${response.status}).`);
        }

        const json = await response.json();
        if (!active) return;

        const safeParse = (val) => {
          if (!val) return [];
          if (typeof val === 'string') {
            try {
              return JSON.parse(val);
            } catch {
              return [];
            }
          }
          return Array.isArray(val) ? val : [];
        };

        const pos = safeParse(json.top_positive_factors);
        const neg = safeParse(json.top_negative_factors);

        // Store back in the patient object locally so it caches and doesn't refetch
        selectedPatient.top_positive_factors = pos;
        selectedPatient.top_negative_factors = neg;

        setTopPositive(pos);
        setTopNegative(neg);
      } catch (err) {
        console.error('SHAP Fetch Error:', err);
        if (active) {
          setShapError('Clinical explainability drivers could not be loaded for this patient.');
        }
      } finally {
        if (active) {
          setShapLoading(false);
        }
      }
    };

    fetchShapExplanation();

    return () => {
      active = false;
    };
  }, [selectedPatient, uploadedFiles]);

  // If no patient is selected, show search input empty state
  if (!selectedPatient) {
    return (
      <div className="max-w-xl mx-auto py-16 text-center space-y-6 animate-fadeIn">
        <div className="w-16 h-16 rounded-2xl bg-sky-500/10 border border-sky-500/30 text-sky-400 flex items-center justify-center mx-auto shadow-inner">
          <User className="w-8 h-8" />
        </div>

        <div className="space-y-2">
          <h3 className="text-xl font-bold text-slate-100">Patient Detail & Clinical Insights</h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Select a patient from High-Risk or Navigation tabs, or search by Member ID below to inspect their risk score, SHAP drivers, and care plan.
          </p>
        </div>

        <form onSubmit={handleSearchSubmit} className="flex gap-2 max-w-md mx-auto">
          <div className="relative flex-1">
            <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Enter Member ID (e.g. M03850)..."
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-sky-500"
            />
          </div>
          <button
            type="submit"
            className="px-5 py-2.5 bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold rounded-xl text-sm transition-colors"
          >
            Search
          </button>
        </form>
      </div>
    );
  }

  const p = selectedPatient;

  // Extract chronic flags for clinical profile badge list
  const chronicList = [
    { label: 'Diabetes', active: p.diabetes === 1 },
    { label: 'COPD', active: p.copd === 1 },
    { label: 'Hypertension', active: p.hypertension === 1 },
    { label: 'CHF', active: p.chf === 1 },
    { label: 'Asthma', active: p.asthma === 1 },
    { label: 'CKD', active: p.ckd === 1 },
  ];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header Info Card */}
      <div className="glass-panel p-6 rounded-2xl flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-sky-500/20 to-indigo-500/20 border border-sky-500/30 text-sky-400 flex items-center justify-center font-mono font-bold text-xl shrink-0">
            {p.member_id.substr(0, 3)}
          </div>
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-extrabold text-slate-100 font-mono tracking-tight">
                {p.member_id}
              </h2>
              <RiskBadge category={p.risk_category} score={p.risk_score} size="lg" />
            </div>

            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-400 font-medium">
              {p.age && <span>Age: <strong className="text-slate-200">{p.age}</strong></span>}
              {p.gender && <span>Gender: <strong className="text-slate-200">{p.gender}</strong></span>}
              <span>Chronic Conditions: <strong className="text-slate-200">{p.num_chronic_conditions}</strong></span>
              <span>Predicted Frequent ED: <strong className={p.predicted_frequent_ED === 1 ? 'text-rose-400' : 'text-emerald-400'}>{p.predicted_frequent_ED === 1 ? 'Yes' : 'No'}</strong></span>
            </div>
          </div>
        </div>

        {/* Large Score Indicator */}
        <div className="flex items-center gap-4 border-t md:border-t-0 md:border-l border-slate-800 pt-4 md:pt-0 md:pl-6 w-full md:w-auto justify-between md:justify-end">
          <div className="text-right">
            <div className="text-xs uppercase tracking-wider text-slate-400 font-semibold">
              Risk Score
            </div>
            <div className="text-4xl font-extrabold font-mono text-slate-100">
              {p.risk_score.toFixed(1)}
              <span className="text-sm font-normal text-slate-400 font-sans"> /100</span>
            </div>
          </div>
        </div>
      </div>

      {/* SAFETY GUARDRAIL BANNER (If Triggered) */}
      {p.safety_guardrail_flag && (
        <SafetyBanner message={p.safety_guardrail_message} />
      )}

      {/* SHAP EXPLAINABILITY DRIVERS */}
      <div className="glass-panel p-6 rounded-2xl relative overflow-hidden">
        {shapLoading ? (
          <div className="flex flex-col items-center justify-center py-12 space-y-3">
            <Loader2 className="w-8 h-8 text-sky-400 animate-spin" />
            <span className="text-xs text-slate-400 font-semibold">Running TreeExplainer inference on backend model...</span>
          </div>
        ) : shapError ? (
          <div className="flex items-center gap-3 p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs">
            <ShieldAlert className="w-5 h-5 text-rose-400 shrink-0" />
            <span>{shapError}</span>
          </div>
        ) : (
          <ShapFactorChart
            topPositive={topPositive}
            topNegative={topNegative}
          />
        )}
      </div>

      {/* CARE NAVIGATION RECOMMENDATIONS (If present) */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
              Care Navigation & Resource Recommendations
            </h3>
            <p className="text-xs text-slate-400">
              Proactive outreach recommendations for non-emergency outpatient support.
            </p>
          </div>
        </div>

        {Array.isArray(p.navigation_opportunities) && p.navigation_opportunities.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {p.navigation_opportunities.map((opp, idx) => (
              <NavigationOpportunityCard key={idx} opportunityText={opp} />
            ))}
          </div>
        ) : (
          <div className="glass-card p-4 rounded-xl text-center text-xs text-slate-400 border-slate-800">
            {p.safety_guardrail_flag
              ? 'Care navigation outreach deferred due to active safety guardrail override (high clinical acuity indicators present).'
              : 'No specific navigation opportunity flags triggered for this patient profile.'}
          </div>
        )}
      </div>

      {/* CLINICAL & ACCESS PROFILE SUMMARY */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Chronic Conditions Profile */}
        <div className="glass-panel p-6 rounded-2xl space-y-4">
          <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Clinical Chronic Condition Flags
          </h4>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
            {chronicList.map((c, idx) => (
              <div
                key={idx}
                className={`p-2.5 rounded-xl border text-xs font-medium flex items-center justify-between ${
                  c.active
                    ? 'bg-rose-500/10 border-rose-500/30 text-rose-300'
                    : 'bg-slate-900/50 border-slate-800 text-slate-500'
                }`}
              >
                <span>{c.label}</span>
                {c.active ? <Check className="w-3.5 h-3.5 text-rose-400" /> : <X className="w-3.5 h-3.5 text-slate-600" />}
              </div>
            ))}
          </div>
        </div>

        {/* Access & Social Determinants Profile */}
        <div className="glass-panel p-6 rounded-2xl space-y-4">
          <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Access Barriers & Care Distance
          </h4>
          <div className="space-y-3 text-xs">
            <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-slate-800">
              <div className="flex items-center gap-2 text-slate-300">
                <Bus className="w-4 h-4 text-purple-400" />
                <span>Transportation Barrier:</span>
              </div>
              <span className={`font-semibold ${p.transportation_barrier === 1 ? 'text-purple-400' : 'text-slate-400'}`}>
                {p.transportation_barrier === 1 ? 'Yes (Barrier Present)' : 'No Barrier'}
              </span>
            </div>

            <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-slate-800">
              <div className="flex items-center gap-2 text-slate-300">
                <PhoneCall className="w-4 h-4 text-emerald-400" />
                <span>Telehealth Available:</span>
              </div>
              <span className={`font-semibold ${p.telehealth_available === 1 ? 'text-emerald-400' : 'text-slate-400'}`}>
                {p.telehealth_available === 1 ? 'Available' : 'Not Available'}
              </span>
            </div>

            <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-slate-800">
              <div className="flex items-center gap-2 text-slate-300">
                <MapPin className="w-4 h-4 text-sky-400" />
                <span>Distance to PCP:</span>
              </div>
              <span className="font-mono text-slate-200 font-semibold">
                {p.pcp_distance_miles !== null ? `${p.pcp_distance_miles} miles` : 'N/A'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
