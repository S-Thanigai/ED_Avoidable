import { useState } from "react";
import { runPrediction } from "./api";
import type { PatientRow, PredictResponse, UploadFiles } from "./types";
import { useTheme } from "./useTheme";
import { Header } from "./components/Header";
import { DisclaimerBanner } from "./components/DisclaimerBanner";
import { UploadPanel } from "./components/UploadPanel";
import { StatCards } from "./components/StatCards";
import { RiskDistributionChart } from "./components/RiskDistributionChart";
import { PatientTable } from "./components/PatientTable";
import { PatientDetailPanel } from "./components/PatientDetailPanel";
import { EmptyState } from "./components/EmptyState";
import { ErrorBanner } from "./components/ErrorBanner";
import { Uc07View } from "./uc07/Uc07View";
import "./App.css";

const EMPTY_FILES: UploadFiles = { members: null, edVisits: null, care: null };

type ActiveTab = "uc07" | "legacy";

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
        (<code>/predict-json</code>). Not the authoritative UC07 multi-agent system. Kept for
        comparison only; its output is never a UC07 risk/navigation/safety decision.
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

function App() {
  const { theme, toggleTheme } = useTheme();
  const [tab, setTab] = useState<ActiveTab>("uc07");

  return (
    <div className="app">
      <Header theme={theme} onToggleTheme={toggleTheme} />

      <nav className="tab-nav" aria-label="Application view">
        <button
          type="button"
          className={`tab-nav__tab${tab === "uc07" ? " tab-nav__tab--active" : ""}`}
          onClick={() => setTab("uc07")}
          aria-current={tab === "uc07" ? "page" : undefined}
        >
          UC07 Navigator
        </button>
        <button
          type="button"
          className={`tab-nav__tab${tab === "legacy" ? " tab-nav__tab--active" : ""}`}
          onClick={() => setTab("legacy")}
          aria-current={tab === "legacy" ? "page" : undefined}
        >
          Legacy Demo
        </button>
      </nav>

      <main className="app__main">{tab === "uc07" ? <Uc07View /> : <LegacyDemoView />}</main>
    </div>
  );
}

export default App;
