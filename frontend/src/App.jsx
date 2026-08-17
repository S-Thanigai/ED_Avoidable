import React, { useState, useMemo } from 'react';
import { Activity, Upload, RefreshCw, Shield, Sparkles, HeartPulse, Info } from 'lucide-react';
import FileUpload from './components/FileUpload';
import TabNav from './components/TabNav';
import PopulationOverview from './tabs/PopulationOverview';
import HighRiskPatients from './tabs/HighRiskPatients';
import NavigationOpportunities from './tabs/NavigationOpportunities';
import PatientDetail from './tabs/PatientDetail';
import TrendsReports from './tabs/TrendsReports';

export default function App() {
  const [patients, setPatients] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState(null);

  const handleDataLoaded = (parsedData, files = null) => {
    setPatients(parsedData);
    setUploadedFiles(files);
    setActiveTab('overview');
    if (parsedData.length > 0) {
      setSelectedPatient(parsedData[0]);
    }
  };

  const handleResetData = () => {
    setPatients(null);
    setUploadedFiles(null);
    setSelectedPatient(null);
    setActiveTab('overview');
  };

  const handleSelectPatient = (patient) => {
    setSelectedPatient(patient);
    setActiveTab('detail');
  };

  // Pre-calculate badge counts for header and navigation tabs
  const highRiskCount = useMemo(() => {
    return patients ? patients.filter(p => p.risk_category === 'High').length : 0;
  }, [patients]);

  const navOppCount = useMemo(() => {
    return patients
      ? patients.filter(
          p => Array.isArray(p.navigation_opportunities) &&
               p.navigation_opportunities.length > 0 &&
               !p.safety_guardrail_flag
        ).length
      : 0;
  }, [patients]);

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 font-sans selection:bg-cyan-500 selection:text-white">
      {/* PERSISTENT HEADER */}
      <header className="border-b border-slate-800/80 bg-slate-900/90 backdrop-blur-xl sticky top-0 z-40 shadow-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          {/* Logo & System Title */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-sky-500 to-indigo-600 flex items-center justify-center text-white shadow-lg shadow-sky-500/20 shrink-0">
              <HeartPulse className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base sm:text-lg font-extrabold tracking-tight text-slate-100 font-mono">
                  ED Risk Navigator
                </h1>
                <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-sky-500/10 text-sky-400 border border-sky-500/20">
                  AI Navigation System
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                Predictive Risk Stratification, SHAP Drivers & Clinical Safety Guardrails
              </p>
            </div>
          </div>

          {/* Action & Status Pill (Visible when data loaded) */}
          {patients && (
            <div className="flex items-center gap-3">
              <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-800/80 border border-slate-700/60 text-xs">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                <span className="text-slate-300 font-medium font-mono">
                  {patients.length.toLocaleString()} Patients Loaded
                </span>
              </div>

              <button
                onClick={handleResetData}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-xs font-semibold transition-colors"
                title="Upload a new scored_patients.csv file"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Load New File</span>
              </button>
            </div>
          )}
        </div>
      </header>

      {/* MAIN CONTENT AREA */}
      {!patients ? (
        <main className="flex-1 flex items-center justify-center">
          <FileUpload onDataLoaded={handleDataLoaded} />
        </main>
      ) : (
        <div className="flex-1 flex flex-col">
          {/* TAB BAR */}
          <TabNav
            activeTab={activeTab}
            onTabChange={setActiveTab}
            highRiskCount={highRiskCount}
            navOppCount={navOppCount}
            selectedPatient={selectedPatient}
          />

          {/* TAB VIEWS */}
          <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6">
            {activeTab === 'overview' && (
              <PopulationOverview patients={patients} />
            )}

            {activeTab === 'high-risk' && (
              <HighRiskPatients
                patients={patients}
                onSelectPatient={handleSelectPatient}
              />
            )}

            {activeTab === 'navigation' && (
              <NavigationOpportunities
                patients={patients}
                onSelectPatient={handleSelectPatient}
              />
            )}

            {activeTab === 'detail' && (
              <PatientDetail
                selectedPatient={selectedPatient}
                patients={patients}
                onSelectPatient={setSelectedPatient}
                uploadedFiles={uploadedFiles}
              />
            )}

            {activeTab === 'trends' && (
              <TrendsReports patients={patients} />
            )}
          </main>
        </div>
      )}

      {/* ETHICAL FOOTER STATEMENT */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-4 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 text-slate-400">
            <Shield className="w-3.5 h-3.5 text-amber-400 shrink-0" />
            <span>
              <strong>Ethical Care Guarantee:</strong> This tool identifies navigation opportunities for lower-acuity care. It never discourages or delays emergency care for patients showing genuine emergency indicators.
            </span>
          </div>

          <div className="text-[11px] text-slate-600 font-mono">
            Vite + React + Tailwind v4 + LightGBM Engine
          </div>
        </div>
      </footer>
    </div>
  );
}
