import type { NavigationDestination } from "./types";

// Single source of truth for how a navigation destination is labeled
// and colored across the app (population chart, results table, member
// workspace) -- so "Urgent Care" reads as the same visual identity
// everywhere instead of three independently-invented mappings. Colors
// reference the --nav-* design tokens (tokens.css); nothing here
// computes or overrides a destination -- it only formats one already
// returned by the backend.
export const NAVIGATION_DESTINATION_LABEL: Record<NavigationDestination, string> = {
  PRIMARY_CARE: "Primary Care",
  URGENT_CARE: "Urgent Care",
  TELEHEALTH: "Telehealth",
  CARE_MANAGEMENT: "Care Management",
  NO_PROACTIVE_NAVIGATION: "No proactive navigation",
};

export const NAVIGATION_DESTINATION_COLOR: Record<NavigationDestination, string> = {
  PRIMARY_CARE: "var(--nav-primary-care)",
  URGENT_CARE: "var(--nav-urgent-care)",
  TELEHEALTH: "var(--nav-telehealth)",
  CARE_MANAGEMENT: "var(--nav-care-management)",
  NO_PROACTIVE_NAVIGATION: "var(--nav-none)",
};

export const NAVIGATION_DESTINATION_ORDER: NavigationDestination[] = [
  "PRIMARY_CARE",
  "URGENT_CARE",
  "TELEHEALTH",
  "CARE_MANAGEMENT",
  "NO_PROACTIVE_NAVIGATION",
];
