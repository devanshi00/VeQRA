import { useState, useEffect, useRef } from "react";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { Card } from "../components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "../components/ui/tabs";
import { AnimatePresence, motion, useScroll, useTransform } from "motion/react";
import {
  Satellite,
  Mail,
  Lock,
  User,
  ArrowRight,
  Sparkles,
} from "lucide-react";
import gsap from "gsap";

/**
 * AuthPage
 * - Login / Signup screen used at app entry.
 * - Handles local form state and basic validation before delegating
 *   authentication events to the parent via onAuth.
 *
 * Props:
 *  - onAuth(action, payload): called with "login" or "signup" and form data
 *  - error: optional error string to display
 *  - setError: setter to update error message
 */
export function AuthPage({ onAuth, error, setError }) {
  // Login form controlled state
  const [loginEmail, setLoginEmail] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  // Signup form controlled state
  const [signupName, setSignupName] = useState("");
  const [signupEmail, setSignupEmail] = useState("");
  const [signupPassword, setSignupPassword] = useState("");
  const [signupConfirmPassword, setSignupConfirmPassword] = useState("");

  // Refs used for background animation targets
  const gridRef = useRef(null);
  const satelliteRef = useRef(null);

  // Small parallax offsets derived from page scroll to animate background decor
  const { scrollY } = useScroll();
  const y1 = useTransform(scrollY, [0, 300], [0, 150]);
  const y2 = useTransform(scrollY, [0, 300], [0, -100]);

  useEffect(() => {
    // Animate subtle drifting grid background
    if (gridRef.current) {
      gsap.to(gridRef.current, {
        backgroundPosition: "200px 200px",
        duration: 20,
        repeat: -1,
        ease: "none",
      });
    }

    // Gentle bobbing animation for the satellite icon block
    if (satelliteRef.current) {
      gsap.to(satelliteRef.current, {
        y: -30,
        duration: 4,
        repeat: -1,
        yoyo: true,
        ease: "power1.inOut",
      });
    }

    // Floating particle animations for decorative dots in the background
    const particles = document.querySelectorAll(".auth-particle");
    particles.forEach((particle, i) => {
      gsap.to(particle, {
        y: -100 - Math.random() * 200,
        x: Math.random() * 100 - 50,
        opacity: 0,
        duration: 3 + Math.random() * 2,
        repeat: -1,
        delay: i * 0.5,
        ease: "power1.out",
      });
    });
    // Run once on mount
  }, []);

  // Handle login form submit: basic client-side validation then delegate
  const handleLogin = (e) => {
    e.preventDefault();
    setError("");
    if (!loginEmail || !loginPassword) {
      setError("Please enter both email and password.");
      return;
    }
    // Parent handles actual auth flow (API calls / navigation)
    onAuth("login", { loginEmail, loginPassword });
  };

  // Handle signup form submit: validate fields and password confirmation
  const handleSignup = (e) => {
    e.preventDefault();
    setError("");
    if (
      !signupName ||
      !signupEmail ||
      !signupPassword ||
      !signupConfirmPassword
    ) {
      setError("Please fill out all fields.");
      return;
    }
    if (signupPassword !== signupConfirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    // Delegate signup to parent
    onAuth("signup", { signupEmail, signupPassword, signupName });
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-linear-to-br from-[#0a1929] via-[#1a2f42] to-[#0d1b2a]">
      {/* Animated grid overlay (decorative) */}
      <div
        ref={gridRef}
        className="absolute inset-0 opacity-10"
        style={{
          backgroundImage: `
            linear-gradient(rgba(72, 202, 228, 0.3) 1px, transparent 1px),
            linear-gradient(90deg, rgba(72, 202, 228, 0.3) 1px, transparent 1px)
          `,
          backgroundSize: "100px 100px",
        }}
      />

      {/* Parallax decorative image blocks that move with scroll */}
      <motion.div
        style={{ y: y1 }}
        className="absolute bottom-0 right-0 opacity-15 blur-sm"
      >
        <div className="absolute inset-0 bg-transparent" />
        <img
          src="/background1.png"
          alt=""
          className="w-full h-full object-cover"
        />
      </motion.div>

      <motion.div
        style={{ y: y2 }}
        className="absolute top-0 left-30 w-[500px] h-[500px] opacity-10 blur-sm"
      >
        <div className="absolute inset-0 bg-transparent" />
        <img
          src="/background2.png"
          alt=""
          className="w-full h-full object-cover"
        />
      </motion.div>

      {/* Floating decorative particles (purely visual) */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(15)].map((_, i) => (
          <div
            key={i}
            className="auth-particle absolute w-1 h-1 bg-[#48cae4] rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              opacity: 0.6,
            }}
          />
        ))}
      </div>

      {/* Subtle blurred color blobs for depth */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-[#0077b6] rounded-full blur-[120px] opacity-20 animate-pulse" />
      <div
        className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-[#48cae4] rounded-full blur-[120px] opacity-20 animate-pulse"
        style={{ animationDelay: "1s" }}
      />

      {/* Main centered content area */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="w-full max-w-md space-y-8"
        >
          <div className="text-center space-y-4">
            {/* Animated Satellite Icon block (brand) */}
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.6 }}
              className="flex justify-center"
            >
              <motion.div
                ref={satelliteRef}
                whileHover={{ scale: 1.1 }}
                className="relative w-20 h-20 rounded-2xl bg-linear-to-br from-[#0077b6] to-[#48cae4] flex items-center justify-center shadow-2xl shadow-[#48cae4]/40"
              >
                <div className="absolute inset-0 rounded-2xl bg-linear-to-br from-[#48cae4]/50 to-transparent blur-xl" />
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{
                    duration: 20,
                    repeat: Infinity,
                    ease: "linear",
                  }}
                >
                  <Satellite className="w-10 h-10 text-white relative z-10" />
                </motion.div>
              </motion.div>
            </motion.div>

            {/* Title and subtitle */}
            <div>
              <h1 className="text-4xl text-white bg-linear-to-r from-white via-[#48cae4] to-white bg-clip-text">
                VeQRA
              </h1>
              <div className="flex items-center justify-center gap-2 mt-2">
                <Sparkles className="w-4 h-4 text-[#48cae4]" />
                <p className="text-[#8d99ae]">
                  Visual Earth Query and Retrieval Assistant
                </p>
              </div>
            </div>
          </div>

          {/* Card containing Login / Signup tabs */}
          <Card className="bg-white/5 border-white/10 backdrop-blur-xl shadow-2xl rounded-2xl overflow-hidden">
            <Tabs defaultValue="login" className="w-full">
              <TabsList className="grid w-full grid-cols-2 bg-white/5 p-1 rounded-t-2xl">
                <TabsTrigger
                  value="login"
                  className="rounded-xl data-[state=active]:bg-linear-to-r data-[state=active]:from-[#0077b6] data-[state=active]:to-[#48cae4] text-white data-[state=active]:shadow-lg transition-all"
                >
                  Login
                </TabsTrigger>
                <TabsTrigger
                  value="signup"
                  className="rounded-xl data-[state=active]:bg-linear-to-r data-[state=active]:from-[#0077b6] data-[state=active]:to-[#48cae4] text-white data-[state=active]:shadow-lg transition-all"
                >
                  Sign Up
                </TabsTrigger>
              </TabsList>

              {/* LOGIN TAB */}
              <TabsContent value="login" className="p-6">
                <form onSubmit={handleLogin} className="space-y-5">
                  <div className="space-y-2">
                    <Label htmlFor="login-email" className="text-white">
                      Email Address
                    </Label>
                    <div className="relative group">
                      {/* Icon inside input for affordance */}
                      <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#8d99ae] group-focus-within:text-[#48cae4] transition-colors" />
                      <Input
                        id="login-email"
                        type="email"
                        placeholder="researcher@example.com"
                        value={loginEmail}
                        onChange={(e) => setLoginEmail(e.target.value)}
                        className="pl-10 bg-white/5 border-white/10 text-white placeholder:text-[#8d99ae] rounded-xl focus:bg-white/10 focus:border-[#48cae4] transition-all"
                        required
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="login-password" className="text-white">
                      Password
                    </Label>
                    <div className="relative group">
                      <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#8d99ae] group-focus-within:text-[#48cae4] transition-colors" />
                      <Input
                        id="login-password"
                        type="password"
                        placeholder="••••••••"
                        value={loginPassword}
                        onChange={(e) => setLoginPassword(e.target.value)}
                        className="pl-10 bg-white/5 border-white/10 text-white rounded-xl focus:bg-white/10 focus:border-[#48cae4] transition-all"
                        required
                      />
                    </div>
                  </div>

                  {/* Small animated error area that appears when `error` prop is set */}
                  <AnimatePresence>
                    {error && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.3 }}
                        className="mt-4 p-3 rounded-xl bg-red-500/50! border border-red-500/30 text-red-400! text-center text-sm"
                      >
                        {error}
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Submit button (sign in) */}
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Button
                      type="submit"
                      className="w-full bg-linear-to-r from-[#0077b6] to-[#48cae4] hover:from-[#0077b6] hover:to-[#00b4d8] text-white rounded-xl py-6 shadow-xl shadow-[#48cae4]/30 transition-all duration-300 relative overflow-hidden group"
                    >
                      <div className="absolute inset-0 bg-linear-to-r from-white/0 via-white/20 to-white/0 translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-1000" />
                      <span className="relative flex items-center justify-center gap-2">
                        Sign In
                        <ArrowRight className="w-4 h-4" />
                      </span>
                    </Button>
                  </motion.div>
                </form>
              </TabsContent>

              {/* SIGNUP TAB */}
              <TabsContent value="signup" className="p-6">
                <form onSubmit={handleSignup} className="space-y-5">
                  <div className="space-y-2">
                    <Label htmlFor="signup-name" className="text-white">
                      Full Name
                    </Label>
                    <div className="relative group">
                      <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#8d99ae] group-focus-within:text-[#48cae4] transition-colors" />
                      <Input
                        id="signup-name"
                        type="text"
                        placeholder="Dr. Jane Smith"
                        value={signupName}
                        onChange={(e) => setSignupName(e.target.value)}
                        className="pl-10 bg-white/5 border-white/10 text-white placeholder:text-[#8d99ae] rounded-xl focus:bg-white/10 focus:border-[#48cae4] transition-all"
                        required
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="signup-email" className="text-white">
                      Email Address
                    </Label>
                    <div className="relative group">
                      <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#8d99ae] group-focus-within:text-[#48cae4] transition-colors" />
                      <Input
                        id="signup-email"
                        type="email"
                        placeholder="researcher@example.com"
                        value={signupEmail}
                        onChange={(e) => setSignupEmail(e.target.value)}
                        className="pl-10 bg-white/5 border-white/10 text-white placeholder:text-[#8d99ae] rounded-xl focus:bg-white/10 focus:border-[#48cae4] transition-all"
                        required
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="signup-password" className="text-white">
                      Password
                    </Label>
                    <div className="relative group">
                      <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#8d99ae] group-focus-within:text-[#48cae4] transition-colors" />
                      <Input
                        id="signup-password"
                        type="password"
                        placeholder="••••••••"
                        value={signupPassword}
                        onChange={(e) => setSignupPassword(e.target.value)}
                        className="pl-10 bg-white/5 border-white/10 text-white rounded-xl focus:bg-white/10 focus:border-[#48cae4] transition-all"
                        required
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label
                      htmlFor="signup-confirm-password"
                      className="text-white"
                    >
                      Confirm Password
                    </Label>
                    <div className="relative group">
                      <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#8d99ae] group-focus-within:text-[#48cae4] transition-colors" />
                      <Input
                        id="signup-confirm-password"
                        type="password"
                        placeholder="••••••••"
                        value={signupConfirmPassword}
                        onChange={(e) =>
                          setSignupConfirmPassword(e.target.value)
                        }
                        className="pl-10 bg-white/5 border-white/10 text-white rounded-xl focus:bg-white/10 focus:border-[#48cae4] transition-all"
                        required
                      />
                    </div>
                  </div>

                  {/* Signup error area (reuses same `error` prop) */}
                  <AnimatePresence>
                    {error && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.3 }}
                        className="mt-4 p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-center text-sm"
                      >
                        {error}
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Create Account button */}
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Button
                      type="submit"
                      className="w-full bg-linear-to-r from-[#0077b6] to-[#48cae4] hover:from-[#0077b6] hover:to-[#00b4d8] text-white rounded-xl py-6 shadow-xl shadow-[#48cae4]/30 transition-all duration-300 relative overflow-hidden group"
                    >
                      <div className="absolute inset-0 bg-linear-to-r from-white/0 via-white/20 to-white/0 translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-1000" />
                      <span className="relative flex items-center justify-center gap-2">
                        Create Account
                        <ArrowRight className="w-4 h-4" />
                      </span>
                    </Button>
                  </motion.div>
                </form>
              </TabsContent>
            </Tabs>
          </Card>

          {/* Small footer note */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
            className="flex items-center justify-center gap-2"
          >
            <div className="w-2 h-2 rounded-full bg-[#48cae4] animate-pulse" />
            <p className="text-center text-sm text-[#8d99ae]">
              Secure authentication for satellite analysis platform
            </p>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}
