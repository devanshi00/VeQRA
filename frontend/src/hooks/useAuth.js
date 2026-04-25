import { useState, useEffect } from "react";

/**
 * useAuth
 * - Small React hook to read persisted login credentials from localStorage.
 * - If credentials are missing or invalid, calls the provided onLogout callback
 *   so the app can redirect the user or perform cleanup.
 *
 * Params:
 *  - onLogout: callback invoked when no valid auth is found (should be stable)
 *
 * Returns:
 *  - { userData, setUserData } where userData is the parsed login credentials or null.
 */
export function useAuth(onLogout) {
  // Local piece of state to hold the parsed user credentials (or null)
  const [userData, setUserData] = useState(null);

  useEffect(() => {
    // Read the stored credentials from localStorage (safely parse or fallback to null)
    const logincred = JSON.parse(localStorage.getItem("logincred") || "null");

    // If no credentials are present, invoke onLogout to handle sign-out flow.
    // Otherwise populate local state so consumers can access user info.
    if (!logincred) onLogout();
    else setUserData(logincred);
  }, [onLogout]); // Re-run if onLogout reference changes

  // Return the current user data and a setter so callers can update state/storage.
  return { userData, setUserData };
}
