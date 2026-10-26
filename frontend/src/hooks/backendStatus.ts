/**
 * The three top-level backend states the whole UI branches on.
 *
 * The distinction that matters is `checking` vs `offline`: a first load shows a
 * neutral "checking" state, while a failed request after that shows the offline
 * banner. Collapsing the two would flash "backend unavailable" on every refresh.
 */
export type BackendStatus = "online" | "offline" | "checking";
