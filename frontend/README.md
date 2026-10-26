# Video Translator — frontend

The React UI for the Video Translator API. It is a single-page app: one build
serves every route, and every route is a real URL (`/jobs/<id>/result`, `/demo`)
so browser back/forward and a shared link both work.

## Development

Two processes, one origin as far as the browser is concerned:

```bash
# terminal 1 — the API
python run.py                 # or: uvicorn src.api.main:app --reload

# terminal 2 — the dev server with hot reload
cd frontend
npm install
npm run dev                   # http://127.0.0.1:5173
```

Open <http://127.0.0.1:5173>. Vite proxies `/api` to the FastAPI process
(`VITE_API_TARGET`, default `http://127.0.0.1:8000`), so the client only ever
talks to one origin and no component needs an absolute backend URL. That is the
same shape production has, which is why nothing in the app changes when the two
are co-located.

## Production

One process serves the API *and* the built UI:

```bash
cd frontend && npm install && npm run build   # writes frontend/dist
python run.py                                 # http://127.0.0.1:8000
```

`frontend/dist` is a build artifact and is not committed (see
`.gitignore`/`frontend/.gitignore`). The server still starts without it — `/`
then explains how to build it — so an API-only deployment is fine.

### F7 notes: how the server serves this app

FastAPI serves the build through `src/api/frontend.py`. The rules, and why:

- **`/api/*`, `/docs`, `/redoc`, `/openapi.json` always belong to the backend.**
  An unmatched `/api/...` path is a JSON 404, never this app's `index.html`.
  Otherwise a typo'd request would return 200 with an HTML body and the client's
  JSON parse would fail with a confusing message instead of a clear "no such
  endpoint".
- **A route this app declares gets `index.html` on a cold load.** Those routes
  are listed once, in `CLIENT_ROUTES` in `src/api/frontend.py`, and mirror the
  `<Route>` table in `src/App.tsx`. A route missing from that list still works
  *within* the app (client-side navigation never asks the server) but a direct
  load or refresh of it would 404 — so when you add a route here, add it there.
- **Anything else is a plain 404.** The fallback is deliberately not a blanket
  "unknown path → shell": that would answer `GET /etc/passwd/media` with 200 and
  an HTML body, and would make a mistyped URL look like a success. Only real
  build files and declared client routes are served.
- **The fallback is middleware, not a catch-all route.** A `@app.get("/{path}")`
  catch-all would make an unmatched path a *partial* match for every other
  method, turning the API's own 404s into 405s and swallowing wrong-method
  responses. Wrapping the finished app instead leaves routing, status codes and
  the error envelope exactly as they were.

The build's hashed assets live under `/assets`, mounted with `StaticFiles` so
they stream and answer byte ranges without this project re-implementing file
serving.

## Tests

```bash
npm test          # vitest
npm run typecheck # tsc -b --noEmit
```

The backend's own coverage of this integration lives in
`tests/test_api_frontend.py`.
