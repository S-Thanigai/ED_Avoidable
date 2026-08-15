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
import "./App.css";

const EMPTY_FILES: UploadFiles = { members: null, edVisits: null, care: null };

function App() {
  const { theme, toggleTheme } = useTheme();
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
    <div className="app">
      <Header theme={theme} onToggleTheme={toggleTheme} />
      <DisclaimerBanner />

      <main className="app__main">
        <UploadPanel
          files={files}
          onFileChange={handleFileChange}
          onRun={handleRun}
          loading={loading}
        />

        {error && (
          <ErrorBanner message={error} onDismiss={() => setError(null)} />
        )}

        {result && result.rows.length > 0 ? (
          <>
            <StatCards rows={result.rows} />
            <RiskDistributionChart rows={result.rows} />
            <PatientTable rows={result.rows} onSelect={setSelected} />
          </>
        ) : (
          !loading && <EmptyState />
        )}
      </main>

      {selected && (
        <PatientDetailPanel row={selected} files={files} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}

export default App;
