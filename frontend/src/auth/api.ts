// Auth API client -- signup/login/logout/me. Every call sends
// `credentials: "include"` so the browser attaches/receives the
// HttpOnly session cookie (backend/auth.py) across the Vite dev
// server's cross-origin request to the FastAPI backend. Never stores a
// token in localStorage/sessionStorage -- the cookie is the only place
// session state lives client-side, and JS cannot read it (HttpOnly).
import { API_BASE_URL } from "../apiConfig";

export interface AuthUser {
  id: number;
  email: string;
}

export class AuthApiError extends Error {
  readonly status: number | null;
  constructor(message: string, status: number | null) {
    super(message);
    this.name = "AuthApiError";
    this.status = status;
  }
}

async function parseErrorDetail(response: Response, fallback: string): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
      if (detail !== undefined) return JSON.stringify(detail);
    }
  } catch {
    /* not JSON -- keep the fallback */
  }
  return fallback;
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { ...init, credentials: "include" });
  } catch {
    throw new AuthApiError(`Could not reach the backend at ${API_BASE_URL}. Is it running?`, null);
  }
  return response;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await request(path, init);
  if (!response.ok) {
    const detail = await parseErrorDetail(response, `Request failed (HTTP ${response.status}).`);
    throw new AuthApiError(detail, response.status);
  }
  return response.json() as Promise<T>;
}

const JSON_HEADERS = { "Content-Type": "application/json" };

export function signup(email: string, password: string): Promise<AuthUser> {
  return requestJson<AuthUser>("/auth/signup", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ email, password }),
  });
}

export function login(email: string, password: string): Promise<AuthUser> {
  return requestJson<AuthUser>("/auth/login", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ email, password }),
  });
}

export async function logout(): Promise<void> {
  await request("/auth/logout", { method: "POST" });
}

/** Returns null (never throws) on 401 -- "not signed in" is an expected,
 * routine state on app load, not an error. */
export async function fetchCurrentUser(): Promise<AuthUser | null> {
  const response = await request("/auth/me", { method: "GET" });
  if (response.status === 401) return null;
  if (!response.ok) {
    const detail = await parseErrorDetail(response, `Request failed (HTTP ${response.status}).`);
    throw new AuthApiError(detail, response.status);
  }
  return response.json() as Promise<AuthUser>;
}
