// Single source of truth for the backend base URL, shared by every API
// client (legacy api.ts, uc07/api.ts, auth/api.ts, populations/api.ts)
// -- one FastAPI app (backend/main.py) serves all of them. Reads
// VITE_API_URL from the environment (see .env.example); falls back to
// the documented local-dev default only, never an Azure/production URL
// (deployment injects the real value later, Phase 9).
//
// MUST share a hostname with the page origin Vite serves (localhost),
// not "127.0.0.1" -- browsers treat "localhost" and "127.0.0.1" as
// DIFFERENT SITES even though both resolve to loopback. The session
// cookie (backend/auth.py) is SameSite=Lax, which is correct for local
// http:// dev, but Lax cookies are only attached to fetch/XHR requests
// that are same-site with the page that sent them. A localhost-page ->
// 127.0.0.1-backend fetch is cross-site, so the browser silently drops
// the cookie on every request after the one that set it -- signup/login
// appears to succeed (Set-Cookie is still accepted+stored), but the
// very next authenticated call 401s. Keeping both sides on "localhost"
// makes them same-site (differing only in port, which same-site
// determination ignores), so the Lax cookie flows normally.
export const API_BASE_URL: string =
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8001";
