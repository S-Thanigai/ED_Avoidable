import React, { useState } from 'react';
import { Upload, FileText, CheckCircle2, AlertCircle, Sparkles, Shield, Compass, Play, FileSpreadsheet } from 'lucide-react';
import { parseScoredPatientsCSV, sanitizePatientRecord } from '../utils/csvParser';

export default function FileUpload({ onDataLoaded }) {
  const [uploadMode, setUploadMode] = useState('raw'); // 'raw' or 'scored'
  const [isHovered, setIsHovered] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // Raw mode file states
  const [membersFile, setMembersFile] = useState(null);
  const [visitsFile, setVisitsFile] = useState(null);
  const [careFile, setCareFile] = useState(null);

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await processFile(file);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsHovered(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    await processFile(file);
  };

  const processFile = async (file) => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const data = await parseScoredPatientsCSV(file);
      onDataLoaded(data, null); // No raw files, only scored CSV
    } catch (err) {
      console.error('CSV Parsing Error:', err);
      setErrorMessage(err.message || 'Failed to parse CSV file. Please check file format.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunModel = async () => {
    if (!membersFile || !visitsFile || !careFile) {
      setErrorMessage('All three CSV files (members, visits, and care history) are required.');
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);
    try {
      const API_BASE_URL = window.location.origin.includes('517') 
        ? 'http://127.0.0.1:8001' 
        : window.location.origin;

      const formData = new FormData();
      formData.append('members_file', membersFile);
      formData.append('ed_visits_file', visitsFile);
      formData.append('care_file', careFile);

      const response = await fetch(`${API_BASE_URL}/predict-json`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let detail = `Prediction request failed (HTTP ${response.status}).`;
        try {
          const err = await response.json();
          if (err?.detail) detail = err.detail;
        } catch {}
        throw new Error(detail);
      }

      const json = await response.json();
      if (json && json.rows) {
        const cleaned = json.rows.map(sanitizePatientRecord);
        // Save the uploaded raw files so detail panel can use them for single patient SHAP on-demand
        onDataLoaded(cleaned, { members: membersFile, edVisits: visitsFile, care: careFile });
        return;
      }
      throw new Error('No prediction output returned from backend model.');
    } catch (err) {
      console.error('Model Execution Error:', err);
      setErrorMessage(err.message || 'Failed to ingest and run backend risk predictions.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadSampleData = async () => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const API_BASE_URL = window.location.origin.includes('517') 
        ? 'http://127.0.0.1:8001' 
        : window.location.origin;

      const response = await fetch(`${API_BASE_URL}/predict-demo`, {
        method: 'POST'
      });

      if (response && response.ok) {
        const json = await response.json();
        if (json && json.rows) {
          const data = json.rows.map(sanitizePatientRecord);
          onDataLoaded(data, null); // Demo mode uses backend local datasets
          return;
        }
      }
      
      throw new Error('Failed to retrieve predictions from backend server.');
    } catch (err) {
      console.error('Demo Ingestion Error:', err);
      setErrorMessage(err.message || 'Could not load demo sample file.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto py-12 px-4 space-y-8">
      {/* Intro Header */}
      <div className="text-center space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-semibold uppercase tracking-wider">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Patient Risk & Navigation System</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-100 tracking-tight">
          Emergency Department Risk Navigator
        </h1>
        <p className="text-slate-400 text-sm sm:text-base max-w-xl mx-auto leading-relaxed">
          Ingest raw clinical datasets to run real-time predictions, or load a pre-scored cohort file to explore AI navigation recommendations.
        </p>
      </div>

      {/* Upload Mode Selector */}
      <div className="flex justify-center border-b border-slate-800/80 max-w-md mx-auto">
        <button
          onClick={() => { setUploadMode('raw'); setErrorMessage(null); }}
          className={`flex-1 pb-3 text-sm font-bold border-b-2 transition-all ${
            uploadMode === 'raw'
              ? 'border-sky-500 text-sky-400'
              : 'border-transparent text-slate-500 hover:text-slate-300'
          }`}
        >
          Ingest Raw Datasets (3 Files)
        </button>
        <button
          onClick={() => { setUploadMode('scored'); setErrorMessage(null); }}
          className={`flex-1 pb-3 text-sm font-bold border-b-2 transition-all ${
            uploadMode === 'scored'
              ? 'border-sky-500 text-sky-400'
              : 'border-transparent text-slate-500 hover:text-slate-300'
          }`}
        >
          Load Scored Output (1 File)
        </button>
      </div>

      {/* Feature Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="glass-card p-4 rounded-xl space-y-2 border-slate-800">
          <div className="p-2 rounded-lg bg-sky-500/10 text-sky-400 w-fit">
            <Sparkles className="w-5 h-5" />
          </div>
          <h3 className="text-sm font-semibold text-slate-200">SHAP Explainability</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Transparent per-patient positive and negative clinical risk drivers.
          </p>
        </div>

        <div className="glass-card p-4 rounded-xl space-y-2 border-slate-800">
          <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400 w-fit">
            <Shield className="w-5 h-5" />
          </div>
          <h3 className="text-sm font-semibold text-slate-200">Safety Guardrails</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Rule-based clinical overrides ensuring high-acuity patients are never redirected.
          </p>
        </div>

        <div className="glass-card p-4 rounded-xl space-y-2 border-slate-800">
          <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400 w-fit">
            <Compass className="w-5 h-5" />
          </div>
          <h3 className="text-sm font-semibold text-slate-200">Proactive Navigation</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Targeted PCP, Telehealth, and Urgent Care opportunities for eligible patients.
          </p>
        </div>
      </div>

      {/* RAW MODE: 3 Files Ingestion */}
      {uploadMode === 'raw' && (
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Slot 1: Members */}
            <div className="p-4 rounded-xl bg-slate-950/40 border border-slate-800 text-center space-y-3 relative hover:border-slate-700 transition-colors">
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setMembersFile(e.target.files?.[0])}
                className="absolute inset-0 opacity-0 cursor-pointer z-10"
                disabled={isLoading}
              />
              <div className="w-10 h-10 rounded-lg bg-sky-500/10 border border-sky-500/20 text-sky-400 flex items-center justify-center mx-auto">
                <FileText className="w-5 h-5" />
              </div>
              <div>
                <span className="text-xs font-bold text-slate-300 block">1. Members CSV</span>
                <span className="text-[10px] text-slate-500">Demographics & chronic history</span>
              </div>
              <div className="text-xs font-medium text-sky-400 truncate px-2">
                {membersFile ? membersFile.name : 'No file selected'}
              </div>
            </div>

            {/* Slot 2: Visits */}
            <div className="p-4 rounded-xl bg-slate-950/40 border border-slate-800 text-center space-y-3 relative hover:border-slate-700 transition-colors">
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setVisitsFile(e.target.files?.[0])}
                className="absolute inset-0 opacity-0 cursor-pointer z-10"
                disabled={isLoading}
              />
              <div className="w-10 h-10 rounded-lg bg-purple-500/10 border border-purple-500/20 text-purple-400 flex items-center justify-center mx-auto">
                <FileSpreadsheet className="w-5 h-5" />
              </div>
              <div>
                <span className="text-xs font-bold text-slate-300 block">2. ED Visits CSV</span>
                <span className="text-[10px] text-slate-500">Dates, diagnoses & triage logs</span>
              </div>
              <div className="text-xs font-medium text-purple-400 truncate px-2">
                {visitsFile ? visitsFile.name : 'No file selected'}
              </div>
            </div>

            {/* Slot 3: Care History */}
            <div className="p-4 rounded-xl bg-slate-950/40 border border-slate-800 text-center space-y-3 relative hover:border-slate-700 transition-colors">
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setCareFile(e.target.files?.[0])}
                className="absolute inset-0 opacity-0 cursor-pointer z-10"
                disabled={isLoading}
              />
              <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center justify-center mx-auto">
                <Compass className="w-5 h-5" />
              </div>
              <div>
                <span className="text-xs font-bold text-slate-300 block">3. Care History CSV</span>
                <span className="text-[10px] text-slate-500">Telehealth, PCP & UC encounters</span>
              </div>
              <div className="text-xs font-medium text-emerald-400 truncate px-2">
                {careFile ? careFile.name : 'No file selected'}
              </div>
            </div>
          </div>

          <button
            onClick={handleRunModel}
            disabled={!membersFile || !visitsFile || !careFile || isLoading}
            className="w-full py-3 bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 disabled:from-slate-800 disabled:to-slate-800 disabled:text-slate-500 text-slate-950 font-bold rounded-xl text-sm transition-all shadow-lg shadow-sky-500/10 flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                <span>Running Ingest & ML Predictions...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span>Execute Model Predictions</span>
              </>
            )}
          </button>
        </div>
      )}

      {/* SCORED MODE: Single CSV Dropzone */}
      {uploadMode === 'scored' && (
        <div
          onDragOver={(e) => { e.preventDefault(); setIsHovered(true); }}
          onDragLeave={() => setIsHovered(false)}
          onDrop={handleDrop}
          className={`glass-panel p-8 sm:p-12 rounded-2xl border-2 border-dashed text-center transition-all duration-300 relative ${
            isHovered
              ? 'border-sky-400 bg-sky-500/10 shadow-2xl shadow-sky-500/20'
              : 'border-slate-700/80 hover:border-slate-600 bg-slate-900/40'
          }`}
        >
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
            disabled={isLoading}
          />

          <div className="space-y-4 pointer-events-none">
            <div className="w-16 h-16 rounded-2xl bg-sky-500/10 border border-sky-500/30 text-sky-400 flex items-center justify-center mx-auto shadow-inner">
              {isLoading ? (
                <div className="w-8 h-8 border-2 border-sky-400 border-t-transparent rounded-full animate-spin" />
              ) : (
                <Upload className="w-8 h-8" />
              )}
            </div>

            <div className="space-y-1">
              <h3 className="text-lg font-bold text-slate-200">
                {isLoading ? 'Processing Scored Patient Data...' : 'Drop scored_patients.csv here'}
              </h3>
              <p className="text-xs text-slate-400">
                or click to browse your local file system
              </p>
            </div>

            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-xs font-mono text-slate-300">
              <FileText className="w-4 h-4 text-sky-400" />
              <span>scored_patients.csv</span>
            </div>
          </div>
        </div>
      )}

      {/* Error Message Display */}
      {errorMessage && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Demo Loader Action */}
      <div className="pt-2 text-center">
        <button
          onClick={handleLoadSampleData}
          disabled={isLoading}
          className="text-xs text-slate-400 hover:text-sky-400 font-medium underline underline-offset-4 transition-colors disabled:opacity-55 disabled:cursor-not-allowed"
        >
          Want to test instantly? Click here to load sample data batch.
        </button>
      </div>
    </div>
  );
}
