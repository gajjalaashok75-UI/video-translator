/**
 * The application shell: skip link, navigation, routed content, footer.
 *
 * The shell itself never blocks on the backend. Individual pages decide how to
 * degrade, so navigation, the language registry and the dashboard remain usable
 * even when one endpoint is failing.
 */

import type { ReactNode } from "react";
import { NavBar } from "./NavBar";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <NavBar />
      <main id="main-content" tabIndex={-1}>
        {children}
      </main>
      <footer className="site-footer">
        <div className="site-footer-inner">
          <span>AI Video Translator — local, self-hosted dubbing.</span>
          <span>React frontend connected to the FastAPI service.</span>
        </div>
      </footer>
    </div>
  );
}
