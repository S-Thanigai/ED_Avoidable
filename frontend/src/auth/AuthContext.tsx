import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { AuthApiError, fetchCurrentUser, login as apiLogin, logout as apiLogout, signup as apiSignup } from "./api";
import type { AuthUser } from "./api";

interface AuthContextValue {
  user: AuthUser | null;
  status: "loading" | "signed-in" | "signed-out";
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

/** Top-level auth state: checks GET /auth/me once on mount to see if an
 * existing session cookie is still valid (e.g. after a page refresh),
 * then exposes login/signup/logout. Nothing about "who is signed in" is
 * ever trusted from anywhere other than the backend's response to these
 * calls -- there is no client-side notion of identity beyond this. */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [status, setStatus] = useState<"loading" | "signed-in" | "signed-out">("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchCurrentUser()
      .then((current) => {
        if (cancelled) return;
        setUser(current);
        setStatus(current ? "signed-in" : "signed-out");
      })
      .catch(() => {
        if (!cancelled) setStatus("signed-out");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  /** Signup/login returning 200/201 only proves the backend accepted the
   * credentials and issued a Set-Cookie header -- it does NOT prove the
   * browser will actually attach that cookie on the NEXT request (e.g.
   * a frontend/backend hostname mismatch causing a same-site cookie to
   * be silently dropped on every subsequent cross-site fetch -- see
   * apiConfig.ts's API_BASE_URL comment). So this app never treats the
   * signup/login response body alone as "signed in": it always follows
   * up with a real GET /auth/me call using the same credentialed
   * client, and only shows a signed-in UI if THAT succeeds. If it
   * doesn't, the user is put back in the signed-out state with a clear
   * error rather than being left showing a signed-in email while every
   * authenticated request silently 401s underneath it. */
  const confirmSession = useCallback(async (expectedUserId: number): Promise<void> => {
    const confirmed = await fetchCurrentUser();
    if (!confirmed || confirmed.id !== expectedUserId) {
      setUser(null);
      setStatus("signed-out");
      throw new AuthApiError(
        "Signed in, but the session could not be confirmed on a follow-up request. " +
          "This usually means the browser is not sending the session cookie back to the API " +
          "(e.g. the frontend and backend are on different hostnames in local development).",
        null,
      );
    }
    setUser(confirmed);
    setStatus("signed-in");
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      setError(null);
      try {
        const created = await apiLogin(email, password);
        await confirmSession(created.id);
      } catch (err) {
        setError(err instanceof AuthApiError || err instanceof Error ? err.message : "Sign in failed.");
        throw err;
      }
    },
    [confirmSession],
  );

  const signup = useCallback(
    async (email: string, password: string) => {
      setError(null);
      try {
        const created = await apiSignup(email, password);
        await confirmSession(created.id);
      } catch (err) {
        setError(err instanceof AuthApiError || err instanceof Error ? err.message : "Sign up failed.");
        throw err;
      }
    },
    [confirmSession],
  );

  const logout = useCallback(async () => {
    await apiLogout();
    setUser(null);
    setStatus("signed-out");
  }, []);

  const clearError = useCallback(() => setError(null), []);

  return (
    <AuthContext.Provider value={{ user, status, error, login, signup, logout, clearError }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth() must be used inside <AuthProvider>.");
  return ctx;
}
