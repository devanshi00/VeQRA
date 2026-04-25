import { useState, useEffect } from "react";
import { Toaster } from "./components/ui/sonner";
import axios from "axios";
import { backendLink } from "../../config";
import { MainInterface } from "./pages/MainInterface";
import { LandingPage } from "./pages/LandingPage";
import { AuthPage } from "./pages/AuthPage";

/**
 * App
 * - Root application component that routes between three top-level screens:
 *   - Landing (marketing) page
 *   - Auth (login / signup) page
 *   - MainInterface (primary logged-in app)
 *
 * Responsibilities:
 *  - Persist / read theme and login state from localStorage
 *  - Perform basic auth requests and store credentials locally
 *  - Wire top-level handlers passed down to child pages
 */
export default function App() {
  // current top-level page ("landing" | "auth" | "interface")
  const [currentPage, setCurrentPage] = useState("landing");
  // UI theme preference persisted to localStorage ("dark" | "light")
  const [theme, setTheme] = useState("dark");
  // display name stored after successful auth
  const [username, setUsername] = useState("");
  // auth error message surface to AuthPage
  const [error, setError] = useState("");

  // On mount: initialize theme from saved preference or system setting
  useEffect(() => {
    const savedTheme = localStorage.getItem("theme");
    const systemTheme = window.matchMedia("(prefers-color-scheme: dark)")
      .matches
      ? "dark"
      : "light";
    const initialTheme = savedTheme || systemTheme;
    setTheme(initialTheme);
    applyTheme(initialTheme);
  }, []);

  // On mount: check for persisted login credentials and restore session
  useEffect(() => {
    const logincred = JSON.parse(localStorage.getItem("logincred") || null);
    if (logincred) {
      // keep the session if last login was within 7 days
      if (
        Date.now() - new Date(logincred.lastLogin) <
        7 * 24 * 60 * 60 * 1000
      ) {
        // refresh lastLogin timestamp and restore username + page
        localStorage.setItem(
          "logincred",
          JSON.stringify({
            lastLogin: Date.now(),
            id: logincred.id,
            username: logincred.username,
            email: logincred.email,
          })
        );
        setUsername(logincred.username || null);
        setCurrentPage("interface");
      } else {
        // expired -> clear stored credentials
        localStorage.removeItem("logincred");
      }
    }
  }, []);

  // Apply theme class to the root element for Tailwind / CSS toggles
  const applyTheme = (newTheme) => {
    if (newTheme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  // Toggle theme and persist choice
  const toggleTheme = () => {
    const newTheme = theme === "light" ? "dark" : "light";
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
    applyTheme(newTheme);
  };

  /**
   * handleAuth
   * - Performs login or signup HTTP requests and persists returned credentials.
   * - On success navigates the user into the main interface and stores username.
   * - On failure surfaces the backend-provided error message to UI.
   *
   * Params:
   *  - type: "login" | anything-else -> treat as signup
   *  - data: form payload (loginEmail/loginPassword or signupName/signupEmail/signupPassword)
   */
  const handleAuth = async (type, data) => {
    try {
      if (type == "login") {
        const res = await axios.post(`${backendLink}/api/auth/login`, {
          email: data.loginEmail,
          password: data.loginPassword,
        });
        if (res.status == 200) {
          // persist minimal credential info for session restore
          localStorage.setItem(
            "logincred",
            JSON.stringify({
              lastLogin: Date.now(),
              id: res.data.id,
              username: res.data.name,
              email: res.data.email,
            })
          );
          setUsername(res.data.name);
          setCurrentPage("interface");
        }
      } else {
        // signup flow
        const res = await axios.post(`${backendLink}/api/auth/signup`, {
          name: data.signupName,
          email: data.signupEmail,
          password: data.signupPassword,
        });
        if (res.status == 200) {
          localStorage.setItem(
            "logincred",
            JSON.stringify({
              lastLogin: Date.now(),
              id: res.data.id,
              username: res.data.name,
              email: res.data.email,
            })
          );
          setUsername(res.data.name);
          setCurrentPage("interface");
        }
      }
    } catch (error) {
      // surface backend error string to the AuthPage for user feedback
      console.log(error);
      setError(error.response.data.error);
    }
  };

  // Clear stored credentials and navigate back to landing page
  const handleLogout = () => {
    localStorage.removeItem("logincred");
    setUsername("");
    setCurrentPage("landing");
  };

  return (
    <>
      {/* Render the current top-level page */}
      {currentPage === "landing" && (
        <LandingPage onGetStarted={() => setCurrentPage("auth")} />
      )}
      {currentPage === "auth" && (
        <AuthPage onAuth={handleAuth} error={error} setError={setError} />
      )}
      {currentPage === "interface" && (
        <MainInterface
          onLogout={handleLogout}
          theme={theme}
          onToggleTheme={toggleTheme}
          username={username}
        />
      )}
      {/* Lightweight global toaster for ephemeral notices */}
      <Toaster />
    </>
  );
}
