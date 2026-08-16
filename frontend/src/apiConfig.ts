// Single source of truth for the backend base URL, shared by both the
// legacy API client (api.ts) and the authoritative UC07 API client
// (uc07/api.ts) -- one FastAPI app (backend/main.py) serves both. Reads
// VITE_API_URL from the environment (see .env.example); falls back to
// the documented local-dev default only, never an Azure/production URL
// (deployment injects the real value later, Phase 9).
export const API_BASE_URL: string =
  (import.meta.env.VITE_API_URL as string | undefined) ?? "http://127.0.0.1:8001";
