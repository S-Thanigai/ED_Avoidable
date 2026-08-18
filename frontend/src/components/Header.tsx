import { useEffect, useState } from "react";
import { getHealth } from "../uc07/api";
import "./Header.css";

interface HeaderProps {
  theme: "light" | "dark";
  onToggleTheme: () => void;
}

type ApiStatus = "checking" | "online" | "unavailable";

/** A real check against GET /health, not a decorative static label --
 * runs once on mount and again every 60s. Never blocks rendering; a
 * failed/slow check just shows "Connection issue", it never throws. */
function useApiStatus(): ApiStatus {
  const [status, setStatus] = useState<ApiStatus>("checking");

  useEffect(() => {
    let cancelled = false;
    const check = () => {
      getHealth()
        .then((health) => {
          if (!cancelled) setStatus(health.status === "ok" ? "online" : "unavailable");
        })
        .catch(() => {
          if (!cancelled) setStatus("unavailable");
        });
    };
    check();
    const interval = window.setInterval(check, 60_000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  return status;
}

const STATUS_LABEL: Record<ApiStatus, string> = {
  checking: "Checking connection…",
  online: "System online",
  unavailable: "Connection issue",
};

export function Header({ theme, onToggleTheme }: HeaderProps) {
  const apiStatus = useApiStatus();
  return (
    <header className="app-header">
      <div className="app-header__inner">
        <div className="app-header__brand">
          <div className="app-header__mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3 4 6.5V12c0 5 3.4 8.7 8 9.9 4.6-1.2 8-4.9 8-9.9V6.5L12 3Z" />
              <path d="M12 8v6M9 11h6" />
            </svg>
          </div>
          <div className="app-header__identity">
            <span className="app-header__wordmark">
              ED <span className="app-header__wordmark-accent">Navigator</span>
            </span>
            <span className="app-header__tagline">Care Management Intelligence</span>
          </div>
        </div>

        <div className="app-header__actions">
          <span
            className={`app-header__status app-header__status--${apiStatus}`}
            title="Backend API connection status"
          >
            <span className="app-header__status-dot" aria-hidden="true" />
            <span className="app-header__status-label">{STATUS_LABEL[apiStatus]}</span>
          </span>
          <button
            type="button"
            className="theme-toggle"
            onClick={onToggleTheme}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
          >
            {theme === "light" ? "🌙" : "☀️"}
            <span>{theme === "light" ? "Dark" : "Light"}</span>
          </button>
        </div>
      </div>
    </header>
  );
}
