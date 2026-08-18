import { useEffect, useState } from "react";
import { useAuth } from "../auth/AuthContext";
import { listPopulations, PopulationsApiError } from "./api";
import type { PopulationSummary } from "./types";
import "./Dashboard.css";

type LoadState = "loading" | "loaded" | "error";

/** Post-login landing screen: Saved Populations + Analyze New CSV.
 * Fetches only population METADATA (id/name/member_count/dates) --
 * never any member-level data -- so this screen stays fast regardless
 * of how large any individual population's member table is. */
export function Dashboard({
  onOpenPopulation,
  onAnalyzeNew,
}: {
  onOpenPopulation: (populationId: number) => void;
  onAnalyzeNew: () => void;
}) {
  const { user, logout } = useAuth();
  const [populations, setPopulations] = useState<PopulationSummary[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setState("loading");
    listPopulations()
      .then((result) => {
        if (cancelled) return;
        setPopulations(result);
        setState("loaded");
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof PopulationsApiError || err instanceof Error ? err.message : "Failed to load.");
        setState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [refreshToken]);

  return (
    <div className="dashboard">
      <div className="dashboard__header">
        <div>
          <h1>Dashboard</h1>
          <p className="dashboard__signed-in-as">Signed in as {user?.email}</p>
        </div>
        <div className="dashboard__header-actions">
          <button type="button" className="dashboard__analyze-btn" onClick={onAnalyzeNew}>
            + Analyze New CSV
          </button>
          <button type="button" className="dashboard__logout-btn" onClick={() => void logout()}>
            Sign out
          </button>
        </div>
      </div>

      <section aria-label="Saved populations">
        <h2 className="dashboard__section-heading">Saved Populations</h2>

        {state === "loading" && <p className="dashboard__status">Loading saved populations…</p>}

        {state === "error" && (
          <div className="dashboard__error" role="alert">
            <span>{error ?? "Database temporarily unavailable."}</span>
            <button type="button" onClick={() => setRefreshToken((t) => t + 1)}>
              Retry
            </button>
          </div>
        )}

        {state === "loaded" && populations.length === 0 && (
          <div className="dashboard__empty">
            <p>No saved populations yet.</p>
            <button type="button" className="dashboard__analyze-btn" onClick={onAnalyzeNew}>
              Analyze CSV
            </button>
          </div>
        )}

        {state === "loaded" && populations.length > 0 && (
          <div className="dashboard__grid">
            {populations.map((population) => (
              <button
                key={population.id}
                type="button"
                className="dashboard__card"
                onClick={() => onOpenPopulation(population.id)}
              >
                <span className="dashboard__card-name">{population.name}</span>
                <span className="dashboard__card-meta">
                  {population.member_count.toLocaleString()} members · analyzed as of {population.index_date}
                </span>
                <span className="dashboard__card-dates">
                  Created {new Date(population.created_at).toLocaleDateString()}
                  {population.updated_at !== population.created_at &&
                    ` · Updated ${new Date(population.updated_at).toLocaleDateString()}`}
                </span>
              </button>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
