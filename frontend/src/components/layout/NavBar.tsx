/**
 * Top navigation.
 *
 * Uses `NavLink` so the current route gets `aria-current="page"` automatically —
 * the active state is exposed to assistive tech, not just colored. On narrow
 * screens the links collapse behind a real `<button>` toggled with
 * `aria-expanded`, and the menu closes on navigation so back/forward always
 * lands on a visible page.
 */

import { useEffect, useRef, useState } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { BackendIndicator } from "../status/BackendIndicator";
import {
  IconActivity,
  IconClose,
  IconFilm,
  IconGlobe,
  IconLayers,
  IconMenu,
  IconSparkle,
} from "../common/Icons";

interface NavItem {
  to: string;
  label: string;
  icon: React.ReactNode;
}

const ITEMS: NavItem[] = [
  { to: "/", label: "Home", icon: <IconFilm size={16} /> },
  { to: "/demo", label: "Demo", icon: <IconSparkle size={16} /> },
  { to: "/dub", label: "Dub", icon: <IconLayers size={16} /> },
  { to: "/jobs", label: "Jobs", icon: <IconActivity size={16} /> },
  { to: "/languages", label: "Languages", icon: <IconGlobe size={16} /> },
  { to: "/dashboard", label: "System", icon: <IconActivity size={16} /> },
];

export function NavBar() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const toggleRef = useRef<HTMLButtonElement>(null);

  // Close the mobile menu whenever the route changes.
  useEffect(() => {
    setOpen(false);
  }, [location.pathname]);

  // Escape closes the menu and returns focus to the toggle.
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
        toggleRef.current?.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <header>
      <nav className="nav" aria-label="Primary">
        <Link to="/" className="nav-brand">
          <span className="nav-brand-mark" aria-hidden="true">
            <IconFilm size={16} />
          </span>
          Video Translator
        </Link>

        <button
          type="button"
          ref={toggleRef}
          className="nav-toggle"
          aria-expanded={open}
          aria-controls="primary-links"
          aria-label={open ? "Close navigation menu" : "Open navigation menu"}
          onClick={() => setOpen((value) => !value)}
        >
          {open ? <IconClose size={18} /> : <IconMenu size={18} />}
        </button>

        <ul id="primary-links" className="nav-links" hidden={!open}>
          {ITEMS.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                end={item.to === "/"}
                className="nav-link"
              >
                {item.icon}
                {item.label}
              </NavLink>
            </li>
          ))}
        </ul>

        <div className="nav-actions">
          <BackendIndicator compact />
        </div>
      </nav>
    </header>
  );
}
