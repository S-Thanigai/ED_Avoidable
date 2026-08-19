import { useState } from "react";
import { runPrediction } from "./api";
import type { PatientRow, PredictResponse, UploadFiles } from "./types";
import { useTheme } from "./useTheme";
import { Header } from "./components/Header";
import { DisclaimerBanner } from "./components/DisclaimerBanner";
import { UploadPanel } from "./components/UploadPanel";
import { StatCards } from "./components/StatCards";
import { RiskDistributionChart } from "./components/RiskDistributionChart";
import { PatientTable } from "./components/LegacyPatientTable";
import { PatientDetailPanel } from "./components/PatientDetailPanel";
import { EmptyState } from "./components/EmptyState";
import { ErrorBanner } from "./components/ErrorBanner";
import { Uc07View } from "./uc07/Uc07View";
import { AuthProvider, useAuth } from "./auth/AuthContext";
import { AuthGate } from "./auth/AuthGate";
import { Dashboard } from "./populations/Dashboard";
import { SavedPopulationView } from "./populations/SavedPopulationView";
import "./App.css";

const EMPTY_FILES: UploadFiles = { members: null, edVisits: null, care: null };

// UC07 (POST /uc07/decide) is the authoritative multi-agent flow and is
// the default tab. The legacy tab is the pre-Phase-2 frequent_ED_user
// model (/predict-json, /explain-member) -- kept available for
// comparison/demo purposes but clearly labeled and never presented as a
// UC07 result (docs/08_FRONTEND_INTEGRATION.md section 21).
function LegacyDemoView() {
  const [files, setFiles] = useState<UploadFiles>(EMPTY_FILES);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [selected, setSelected] = useState<PatientRow | null>(null);

  const handleFileChange = (key: keyof UploadFiles, file: File | null) => {
    setFiles((prev) => ({ ...prev, [key]: file }));
  };

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await runPrediction(files);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="legacy-banner" role="note">
        <strong>Legacy demo model</strong> — pre-Phase-2 <code>frequent_ED_user</code> model
        (<code>/predict-json</code>). Not the authoritative ED Navigator multi-agent system. Kept for
        comparison only; its output is never an ED Navigator risk/navigation/safety decision.
      </div>
      <DisclaimerBanner />

      <UploadPanel files={files} onFileChange={handleFileChange} onRun={handleRun} loading={loading} />

      {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

      {result && result.rows.length > 0 ? (
        <>
          <StatCards rows={result.rows} />
          <RiskDistributionChart rows={result.rows} />
          <PatientTable rows={result.rows} onSelect={setSelected} />
        </>
      ) : (
        !loading && <EmptyState />
      )}

      {selected && (
        <PatientDetailPanel row={selected} files={files} onClose={() => setSelected(null)} />
      )}
    </>
  );
}

type Screen = "dashboard" | "analyze" | "legacy" | "population";

/** Post-authentication app shell: Dashboard (Saved Populations + Analyze
 * New CSV) is the landing screen; "Analyze New CSV" opens the EXISTING,
 * untouched Uc07View CSV flow; opening a saved population shows
 * SavedPopulationView instead (server-paginated, no CSV re-upload). The
 * Legacy Demo tab is unchanged and still available. */
function AuthenticatedApp() {
  const { theme, toggleTheme } = useTheme();
  const [screen, setScreen] = useState<Screen>("dashboard");
  const [activePopulationId, setActivePopulationId] = useState<number | null>(null);

  const openPopulation = (id: number) => {
    setActivePopulationId(id);
    setScreen("population");
  };

  const backToDashboard = () => {
    setActivePopulationId(null);
    setScreen("dashboard");
  };

  if (screen === "population" && activePopulationId !== null) {
    return (
      <div className="app">
        <Header theme={theme} onToggleTheme={toggleTheme} />
        <main className="app__main">
          <SavedPopulationView populationId={activePopulationId} onBack={backToDashboard} onDeleted={backToDashboard} />
        </main>
      </div>
    );
  }

  return (
    <div className="app">
      <Header theme={theme} onToggleTheme={toggleTheme} />

      <nav className="tab-nav" aria-label="Application view">
        <button
          type="button"
          className={`tab-nav__tab tab-nav__tab--primary${screen === "dashboard" ? " tab-nav__tab--active" : ""}`}
          onClick={() => setScreen("dashboard")}
          aria-current={screen === "dashboard" ? "page" : undefined}
        >
          Dashboard
        </button>
        <button
          type="button"
          className={`tab-nav__tab tab-nav__tab--primary${screen === "analyze" ? " tab-nav__tab--active" : ""}`}
          onClick={() => setScreen("analyze")}
          aria-current={screen === "analyze" ? "page" : undefined}
        >
          Analyze CSV
        </button>

      </nav>

      <main className="app__main">
        {screen === "dashboard" && (
          <Dashboard onOpenPopulation={openPopulation} onAnalyzeNew={() => setScreen("analyze")} />
        )}
        {screen === "analyze" && <Uc07View />}
        {screen === "legacy" && <LegacyDemoView />}
      </main>
    </div>
  );
}

function AppShell() {
  const { status } = useAuth();

  if (status === "loading") {
    return (
      <div className="app-loading" role="status">
        Loading…
      </div>
    );
  }

  if (status === "signed-out") {
    return <AuthGate />;
  }

  return <AuthenticatedApp />;
}

function App() {
  return (
    <AuthProvider>
      <AppShell />
    </AuthProvider>
  );
}

export default App;
