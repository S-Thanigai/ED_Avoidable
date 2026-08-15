# ED Navigator Dashboard (React)

React + TypeScript + Vite frontend for the Avoidable ED Utilization Navigator.

## Develop

```bash
npm install
npm run dev
```

Runs on `http://localhost:5173` and calls the FastAPI backend directly. By
default it targets `http://127.0.0.1:8001` — override with a `.env` file
(see `.env.example`):

```
VITE_API_URL=http://127.0.0.1:8001
```

## Build for production

```bash
npm run build
```

Outputs to `dist/`. The FastAPI backend (`backend/main.py`) serves this
folder at `/dashboard` automatically once it exists — run the backend after
building, or re-run `npm run build` after frontend changes to update it.
