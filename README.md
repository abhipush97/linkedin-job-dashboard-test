# Jobs Metasearch (Skyscanner-Style)

This project is now refactored to a modern product stack:

- **Backend**: FastAPI API layer on top of your existing multi-source job engine
- **Frontend**: React 19 + TypeScript + Vite + React Query
- **UX direction**: Skyscanner-style journey for jobs (search-first, compare fast, refine with confidence)
- **Auth**: basic username/password login (session token)

## Product flow

1. Sign in
2. Run one search across selected sources
3. See live source health + coverage
4. Refine with quick filters (text/company/recency/remote/external links)
5. Open apply links directly from results cards

## Workspace pages

- `/search`: query setup, source selection, and advanced source options
- `/results`: filters + paginated result cards (not one long list)
- `/sources`: source reliability, run reports, and event timeline

## Tech stack

- FastAPI, Uvicorn, Pandas, Requests, BeautifulSoup
- React 19, TypeScript, Vite, TanStack React Query

## Setup

```bash
cd /Users/abhishekpushkarjha/Desktop/Organized_Files/2026-02\ -\ February\ \(4.09\ GB\)/linkedin
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Frontend dependencies:

```bash
cd web
npm install
```

## Run backend (API)

From project root:

```bash
.venv/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Run frontend

In a second terminal:

```bash
cd web
npm run dev
```

Open: `http://localhost:5173`

## Login

Default credentials:

- Username: `recruiter`
- Password: `linkedin123`

Override with environment variables:

```bash
export APP_LOGIN_USERNAME="your_username"
export APP_LOGIN_PASSWORD="your_password"
```

Optional token/session settings:

```bash
export APP_TOKEN_TTL_SECONDS=43200
```

## Publish for alpha testing

### Option A (recommended): Vercel frontend + Render backend

1. Deploy backend (`FastAPI`) as a web service:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
2. Set backend env vars:
   - `APP_LOGIN_USERNAME`
   - `APP_LOGIN_PASSWORD`
   - `APP_CORS_ORIGINS` (include your frontend URL)
3. Deploy frontend from `web/`:
   - Build command: `npm run build`
   - Output directory: `dist`
   - Env var: `VITE_API_BASE_URL=https://your-backend-url`
4. `web/vercel.json` is included so React routes (`/search`, `/results`, `/sources`) rewrite to `index.html`.

### Option B: Railway (frontend + backend as separate services)

- Same env vars and commands as above.
- Point frontend `VITE_API_BASE_URL` to deployed backend URL.

## API overview

- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/auth/me`
- `GET /api/sources`
- `POST /api/search`
- `GET /api/health`

## Notes

- Existing source adapters and search logic remain in `job_sources.py`.
- Streamlit app file is kept for backward compatibility but is no longer the primary UI.
- Some sources may return blocked/empty depending on anti-bot restrictions or geography.
