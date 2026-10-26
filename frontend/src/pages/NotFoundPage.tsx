/**
 * 404 page. Reachable by any unknown path, including a stale bookmark.
 */

import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <div className="page">
      <div className="state-block">
        <span className="state-title">Page not found</span>
        <span className="state-body">
          The page you asked for does not exist. It may have moved, or the link
          may be out of date.
        </span>
        <Link to="/" className="btn btn-primary">
          Back to home
        </Link>
      </div>
    </div>
  );
}
