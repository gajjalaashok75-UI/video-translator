/**
 * Routing.
 *
 * Real client-side routes with `BrowserRouter`, so browser back/forward and
 * deep links work. `/demo` is the guided demo (F6).
 *
 * In production FastAPI must serve `index.html` for any non-`/api` path so a
 * deep link reaches this router (see the F7 notes in the frontend README).
 */

import { BrowserRouter, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { BackendHealthProvider } from "./hooks/useHealth";
import { LandingPage } from "./pages/LandingPage";
import { DashboardPage } from "./pages/DashboardPage";
import { LanguageExplorerPage } from "./pages/LanguageExplorerPage";
import { JobsPage } from "./pages/JobsPage";
import { DubPage } from "./pages/DubPage";
import { DemoPage } from "./pages/DemoPage";
import { JobDetailPage } from "./pages/JobDetailPage";
import { JobResultPage } from "./pages/JobResultPage";
import { NotFoundPage } from "./pages/NotFoundPage";

export function App() {
  return (
    <BackendHealthProvider>
      <BrowserRouter>
        <AppShell>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/languages" element={<LanguageExplorerPage />} />
            <Route path="/jobs" element={<JobsPage />} />
            <Route path="/jobs/:jobId" element={<JobDetailPage />} />
            <Route path="/jobs/:jobId/result" element={<JobResultPage />} />
            <Route path="/dub" element={<DubPage />} />
            <Route path="/demo" element={<DemoPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </AppShell>
      </BrowserRouter>
    </BackendHealthProvider>
  );
}
