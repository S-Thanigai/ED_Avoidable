import { useState } from "react";
import type { FormEvent } from "react";
import { useAuth } from "./AuthContext";
import "./AuthGate.css";

type Mode = "signin" | "signup";

/** The unauthenticated entry screen -- toggles between Sign In and Sign
 * Up. Rendered by App.tsx whenever auth status is "signed-out"; nothing
 * behind it (dashboard, CSV analysis) is reachable without a valid
 * session, since every data-bearing request depends on the session
 * cookie anyway (the backend, not this component, is the real gate). */
export function AuthGate() {
  const { login, signup, error, clearError } = useAuth();
  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const switchMode = (next: Mode) => {
    setMode(next);
    clearError();
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      if (mode === "signin") {
        await login(email, password);
      } else {
        await signup(email, password);
      }
    } catch {
      /* error already captured in context state */
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-gate">
      <form className="auth-gate__card" onSubmit={handleSubmit}>
        <div className="auth-gate__brand">
          <div className="auth-gate__mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3 4 6.5V12c0 5 3.4 8.7 8 9.9 4.6-1.2 8-4.9 8-9.9V6.5L12 3Z" />
              <path d="M12 8v6M9 11h6" />
            </svg>
          </div>
          <span className="auth-gate__wordmark">ED Navigator</span>
        </div>

        <h1 className="auth-gate__heading">{mode === "signin" ? "Sign in" : "Create an account"}</h1>
        <p className="auth-gate__subheading">
          {mode === "signin"
            ? "Sign in to view your saved populations or analyze a new CSV."
            : "Save analyzed populations to your account so you don't have to re-upload CSVs."}
        </p>

        {error && (
          <div className="auth-gate__error" role="alert">
            {error}
          </div>
        )}

        <label className="auth-gate__field">
          <span>Email</span>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
            required
            disabled={submitting}
          />
        </label>

        <label className="auth-gate__field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={mode === "signin" ? "current-password" : "new-password"}
            minLength={mode === "signup" ? 8 : undefined}
            required
            disabled={submitting}
          />
          {mode === "signup" && <span className="auth-gate__hint">At least 8 characters.</span>}
        </label>

        <button type="submit" className="auth-gate__submit" disabled={submitting}>
          {submitting ? "Please wait…" : mode === "signin" ? "Sign in" : "Create account"}
        </button>

        <p className="auth-gate__switch">
          {mode === "signin" ? (
            <>
              Don't have an account?{" "}
              <button type="button" onClick={() => switchMode("signup")}>
                Sign up
              </button>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <button type="button" onClick={() => switchMode("signin")}>
                Sign in
              </button>
            </>
          )}
        </p>
      </form>
    </div>
  );
}
