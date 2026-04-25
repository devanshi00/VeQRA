import { motion } from "motion/react";
import { Button } from "./ui/button";
import { Menu, Moon, Sun, LogOut, FileText, User } from "lucide-react";

/**
 * Header
 * - Top navigation bar used across the app.
 * - Props:
 *   - isDark: boolean theme flag
 *   - username: string to show current user
 *   - uploadedImage: optional object; presence enables "Query Image" action
 *   - onToggleHistory: open/close chat history panel
 *   - onToggleTheme: toggle light/dark theme
 *   - onOpenQuerySidebar: open the query sidebar (only relevant if uploadedImage exists)
 *   - onLogout: sign out callback
 */
export function Header({
  isDark,
  username,
  uploadedImage,
  onToggleHistory,
  onToggleTheme,
  onOpenQuerySidebar,
  onLogout,
}) {
  return (
    <header
      // sticky top bar with subtle backdrop blur and theme-aware border/background
      className={`sticky top-0 z-30 border-b backdrop-blur-xl transition-all duration-300 ${
        isDark
          ? "border-white/10 bg-[#1b263b]/80"
          : "border-[#e3edf3] bg-white/80"
      }`}
    >
      <div className="container mx-auto px-2 md:px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Left group: menu toggle and brand */}
          <div className="flex items-center gap-4">
            <motion.div>
              {/* Button to toggle the chat history slide-over */}
              <Button
                onClick={onToggleHistory}
                variant="ghost"
                size="icon"
                className={`rounded-xl shadow-xl transition-all duration-300 ${
                  isDark
                    ? "bg-white/10 backdrop-blur-xl border border-white/20 hover:bg-white/20 text-white"
                    : "bg-white border border-[#e3edf3] hover:bg-[#e3edf3] shadow-lg"
                }`}
              >
                <Menu className="w-5 h-5" />
              </Button>
            </motion.div>

            {/* Brand: logo + name (hidden on very small screens) */}
            <div className="hidden sm:flex items-center gap-2">
              <motion.div
                transition={{ duration: 0.6 }}
                className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                  isDark
                    ? "bg-linear-to-br from-[#48cae4] to-[#0077b6]"
                    : "bg-linear-to-br from-[#0077b6] to-[#005f8f]"
                }`}
              >
                <img src="/logo.png" alt="Logo" className="rounded-lg" />
              </motion.div>
              <span className={`${isDark ? "text-white" : "text-[#1b263b]"}`}>
                VeQRA
              </span>
            </div>
          </div>

          {/* Right group: user info, theme toggle, query action and logout */}
          <div className="flex items-center gap-3">
            {/* Compact user badge (hidden on smallest screens) */}
            <div
              className={`hidden sm:flex items-center gap-2 px-3 py-2 rounded-lg backdrop-blur-sm ${
                isDark ? "bg-white/10" : "bg-[#e3edf3]"
              }`}
            >
              <User
                className={`w-4 h-4 ${
                  isDark ? "text-[#48cae4]" : "text-[#0077b6]"
                }`}
              />
              <span
                className={`text-sm ${
                  isDark ? "text-white" : "text-[#1b263b]"
                }`}
              >
                {username}
              </span>
            </div>

            {/* Theme toggle: shows sun when dark (click to switch), moon when light */}
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button
                variant="ghost"
                size="icon"
                onClick={onToggleTheme}
                className={`rounded-lg transition-all ${
                  isDark ? "hover:bg-white/10" : "hover:bg-[#e3edf3]"
                }`}
              >
                {isDark ? (
                  // Display Sun icon in dark mode (indicates switching to light)
                  <Sun className="w-5 h-5 text-white" />
                ) : (
                  // Display Moon icon in light mode (indicates switching to dark)
                  <Moon className="w-5 h-5" />
                )}
              </Button>
            </motion.div>

            {/* Query Image button: only shown if an image is uploaded */}
            {uploadedImage && (
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  onClick={onOpenQuerySidebar}
                  className={`rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 ${
                    isDark
                      ? "bg-linear-to-r from-[#48cae4] to-[#00b4d8] hover:from-[#48cae4] hover:to-[#0096c7] text-[#0d1b2a]"
                      : "bg-linear-to-r from-[#0077b6] to-[#005f8f] hover:from-[#0077b6] hover:to-[#0099cc] text-white"
                  }`}
                >
                  <Menu className="w-4 h-4 mr-2" />
                  Query Image
                </Button>
              </motion.div>
            )}

            {/* Logout button: small icon-only control */}
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button
                variant="ghost"
                size="icon"
                onClick={onLogout}
                className="rounded-lg hover:bg-red-50 dark:text-white dark:hover:bg-red-950/20 hover:text-[#c1121f] transition-all"
              >
                <LogOut className={`w-5 h-5`} />
              </Button>
            </motion.div>
          </div>
        </div>
      </div>
    </header>
  );
}
