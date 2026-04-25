import { useEffect, useRef } from "react";
import { Button } from "../components/ui/button";
import { motion, useScroll, useTransform } from "motion/react";
import {
  Satellite,
  Globe,
  ArrowRight,
  Sparkles,
  Brain,
  Mic,
  MessageSquare,
  Image as ImageIcon,
  CheckCircle2,
} from "lucide-react";
import gsap from "gsap";
import { ImageWithFallback } from "../components/ImageWithFallback";
import { Card } from "../components/ui/card";

/**
 * LandingPage
 * - Marketing / home page shown before users enter the main app.
 * - Displays hero, feature highlights and example outputs with decorative
 *   animated background elements.
 *
 * Props:
 *  - onGetStarted: callback invoked when the primary CTA is clicked
 */
export function LandingPage({ onGetStarted }) {
  // refs for DOM nodes used by GSAP animations and decorative layout
  const containerRef = useRef(null);
  const earthRef = useRef(null); // rotates and bobs slightly
  const satelliteRef = useRef(null); // subtle bobbing
  const gridRef = useRef(null); // animated grid background

  // scroll-driven parallax values (motion/react hooks)
  const { scrollY } = useScroll();
  const y1 = useTransform(scrollY, [0, 300], [0, 150]); // used for right-side decor
  const y2 = useTransform(scrollY, [0, 300], [0, -100]); // used for left-side decor

  useEffect(() => {
    // Animate tiled grid background to slowly pan (decorative)
    if (gridRef.current) {
      gsap.to(gridRef.current, {
        backgroundPosition: "200px 200px",
        duration: 20,
        repeat: -1,
        ease: "none",
      });
    }

    // Make the Earth block gently bob and rotate for subtle motion
    if (earthRef.current) {
      gsap.to(earthRef.current, {
        y: -30,
        duration: 4,
        repeat: -1,
        yoyo: true,
        ease: "power1.inOut",
      });
      gsap.to(earthRef.current, {
        rotation: 360,
        duration: 60,
        repeat: -1,
        ease: "none",
      });
    }

    // Small bob for the satellite block to add variety
    if (satelliteRef.current) {
      gsap.to(satelliteRef.current, {
        y: 20,
        duration: 3,
        repeat: -1,
        yoyo: true,
        ease: "power1.inOut",
      });
    }

    // Floating particle decorations that drift upwards repeatedly
    const particles = document.querySelectorAll(".particle");
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

  return (
    <div
      ref={containerRef}
      className="relative min-h-screen overflow-auto bg-linear-to-br from-[#0a1929] via-[#1a2f42] to-[#0d1b2a]"
    >
      {/* Tiled grid background used by GSAP to create a slow drifting effect */}
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

      {/* Parallax decorative blocks that move with scroll */}
      <motion.div
        style={{ y: y1 }}
        className="absolute top-0 right-0 w-[800px] h-[800px] opacity-20 blur-sm"
      >
        <div className="absolute inset-0 bg-linear-to-l from-transparent via-transparent to-[#0a1929]" />
      </motion.div>

      <motion.div
        style={{ y: y2 }}
        className="absolute bottom-0 left-0 w-[600px] h-[600px] opacity-15 blur-sm"
      >
        <div className="absolute inset-0 bg-linear-to-r from-transparent via-transparent to-[#0a1929]" />
      </motion.div>

      {/* Animated particle elements (purely visual) */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="particle absolute w-1 h-1 bg-[#48cae4] rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              opacity: 0.6,
            }}
          />
        ))}
      </div>

      {/* Decorative rotating Earth icon (hidden on small screens) */}
      <div
        ref={earthRef}
        className="absolute top-32 right-20 w-32 h-32 opacity-20 pointer-events-none hidden lg:block"
      >
        <Satellite className="w-full h-full text-[#48cae4]" strokeWidth={0.5} />
      </div>

      {/* Decorative globe icon that bobs, for depth */}
      <div
        ref={satelliteRef}
        className="absolute top-32 left-20 w-32 h-32 opacity-20 pointer-events-none hidden lg:block"
      >
        <Globe className="w-full h-full text-[#48cae4]" strokeWidth={0.5} />
      </div>

      {/* Soft blurred color blobs for depth */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-[#0077b6] rounded-full blur-[120px] opacity-20 animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-[#48cae4] rounded-full blur-[120px] opacity-20 animate-pulse delay-1000" />

      {/* Hero / main content */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: "easeOut" }}
          className="max-w-7xl mx-auto text-center space-y-6"
        >
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.7, duration: 0.8 }}
            className="flex flex-col lg:flex-row justify-center items-center gap-4"
          >
            <div className="flex flex-col gap-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5, duration: 0.8 }}
                className="space-y-4"
              >
                {/* Small badge describing the product */}
                <div className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-linear-to-r from-[#48cae4]/20 to-[#0077b6]/20 backdrop-blur-sm border border-[#48cae4]/30">
                  <Sparkles className="w-4 h-4 text-[#48cae4]" />
                  <span className="text-sm text-[#48cae4]">
                    Visual Earth Query and Retrieval Assistant
                  </span>
                </div>
              </motion.div>

              {/* Headline */}
              <h1 className="text-white text-6xl md:text-7xl lg:text-8xl tracking-tight">
                <span className="inline-block bg-linear-to-r from-white via-[#48cae4] to-white bg-clip-text text-transparent animate-gradient">
                  VeQRA
                </span>
              </h1>

              {/* Short descriptive paragraph highlighting value props */}
              <motion.p className="text-xl text-[#b9d6f2] max-w-3xl mx-auto leading-relaxed">
                Gain{" "}
                <span className="text-[#48cae4] font-medium">
                  advanced insights
                </span>{" "}
                from{" "}
                <span className="text-[#48cae4] font-medium">
                  satellite imagery
                </span>{" "}
                for comprehensive land, water, and vegetation analysis. Harness
                the power of cutting-edge AI to explore Earth observation data
                effortlessly through{" "}
                <span className="text-[#48cae4] font-medium">
                  intuitive visual queries
                </span>
                ,{" "}
                <span className="text-[#48cae4] font-medium">
                  intelligent captioning
                </span>
                , and{" "}
                <span className="text-[#48cae4] font-medium">
                  automated interpretation tools
                </span>
                . From detecting subtle environmental changes to understanding
                complex geographic patterns, experience a smarter, more
                interactive way to analyze our planet — all within a sleek,
                modern interface .
              </motion.p>

              {/* Primary CTA */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.9, duration: 0.8 }}
                className="pt-6"
              >
                <Button
                  onClick={onGetStarted}
                  className="group relative px-10 py-8 text-lg bg-linear-to-r from-[#0077b6] to-[#005f8f] hover:from-[#0077b6] hover:to-[#0099cc] text-white rounded-2xl shadow-2xl shadow-[#0077b6]/40 hover:shadow-[#0077b6]/60 transition-all duration-500 overflow-hidden"
                >
                  <div className="absolute inset-0 bg-linear-to-r from-white/0 via-white/20 to-white/0 translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-1000" />
                  <span className="relative flex items-center gap-3">
                    Get Started
                    <motion.span
                      animate={{ x: [0, 5, 0] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                    >
                      <ArrowRight className="w-5 h-5" />
                    </motion.span>
                  </span>
                </Button>
              </motion.div>
            </div>

            {/* Brand / illustrative image */}
            <img
              src="/logo_2.png"
              alt="Logo"
              className="rounded-lg w-100 h-100"
            />
          </motion.div>
        </motion.div>
      </div>

      {/* Example outputs section: showcases sample queries + results */}
      <div className="relative z-10 px-6 py-24">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-linear-to-r from-[#48cae4]/20 to-[#0077b6]/20 backdrop-blur-sm border border-[#48cae4]/30 mb-6">
              <ImageIcon className="w-4 h-4 text-[#48cae4]" />
              <span className="text-sm text-[#48cae4]">Example Outputs</span>
            </div>
            <h2 className="text-white text-5xl mb-4">See It In Action</h2>
            <p className="text-[#8d99ae] text-lg max-w-2xl mx-auto">
              Real examples of satellite imagery analysis powered by our VQA
              model
            </p>
          </motion.div>

          {/* Grid of example cards (image + query + sample result) */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {[
              {
                image: "/display 1.png",
                query:
                  "How many aircraft are visible on the ground in this image?",
                result: "Two aircraft are visible parked near the taxiway.",
              },
              {
                image: "/display 2.png",
                query:
                  "Which side of the image appears to contain more residential buildings — left or right?",
                result:
                  "The left side of the image has a significantly higher density of residential buildings.",
              },
              {
                image: "/display 3.png",
                query:
                  "Are the aircraft in this image parked or preparing for takeoff?",
                result:
                  "The aircraft appear to be parked on the apron, not aligned with the runway for takeoff.",
              },
              {
                image: "/display 4.png",
                query:
                  "What recreational facilities can you identify in this image?",
                result:
                  "The image shows a tennis court and a swimming pool within a residential property.",
              },
            ].map((example, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1, duration: 0.6 }}
                whileHover={{ y: -8, transition: { duration: 0.3 } }}
              >
                <Card className="overflow-hidden bg-linear-to-br from-white/10 to-white/5 backdrop-blur-xl border border-white/20 hover:border-white/40 transition-all duration-500">
                  {/* Example image uses ImageWithFallback for robust loading */}
                  <div className="relative h-64 overflow-hidden">
                    <ImageWithFallback
                      src={example.image}
                      alt={`Example ${i + 1}`}
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute inset-0 bg-linear-to-t from-[#0d1b2a] via-transparent to-transparent" />
                  </div>

                  {/* Query + sample result */}
                  <div className="p-6 space-y-4">
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 rounded-lg bg-[#48cae4]/20 flex items-center justify-center shrink-0">
                        <MessageSquare className="w-4 h-4 text-[#48cae4]" />
                      </div>
                      <p className="text-sm text-[#b9d6f2]">{example.query}</p>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 rounded-lg bg-[#52b788]/20 flex items-center justify-center shrink-0">
                        <CheckCircle2 className="w-4 h-4 text-[#52b788]" />
                      </div>
                      <p className="text-sm text-[#8d99ae] leading-relaxed">
                        {example.result}
                      </p>
                    </div>
                  </div>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Features section describing AI capabilities */}
      <div className="relative z-10 px-6 py-24 bg-linear-to-b from-transparent via-[#0d1b2a]/50 to-[#0a1929]">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-linear-to-r from-[#48cae4]/20 to-[#0077b6]/20 backdrop-blur-sm border border-[#48cae4]/30 mb-6">
              <Brain className="w-4 h-4 text-[#48cae4]" />
              <span className="text-sm text-[#48cae4]">
                Advanced AI Features
              </span>
            </div>
            <h2 className="text-white text-5xl mb-4">
              Powered by Intelligence
            </h2>
            <p className="text-[#8d99ae] text-lg max-w-2xl mx-auto">
              Cutting-edge AI capabilities that make satellite analysis
              intuitive and powerful
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Feature card: LLM integration */}
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
            >
              <Card className="p-8 bg-linear-to-br from-white/10 to-white/5 backdrop-blur-xl border border-white/20 hover:border-[#48cae4]/40 transition-all duration-500 h-full group">
                <div className="absolute inset-0 bg-linear-to-br from-[#48cae4]/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-xl" />
                <div className="relative">
                  <div className="w-20 h-20 rounded-2xl bg-linear-to-br from-[#0077b6]/30 to-[#48cae4]/30 flex items-center justify-center mb-6">
                    <Brain className="w-10 h-10 text-[#48cae4]" />
                  </div>
                  <h3 className="text-white text-2xl mb-4">
                    Large Language Model Integration
                  </h3>
                  <p className="text-[#8d99ae] leading-relaxed mb-6">
                    Our platform leverages state-of-the-art LLMs fine-tuned for
                    geospatial analysis. This enables natural language
                    understanding of complex queries and generation of detailed,
                    contextually aware responses about satellite imagery.
                  </p>
                  <div className="space-y-3">
                    {[
                      "Natural language query processing",
                      "Context-aware response generation",
                      "Multi-turn conversation support",
                      "Domain-specific knowledge integration",
                    ].map((feature, i) => (
                      <div key={i} className="flex items-center gap-3">
                        <CheckCircle2 className="w-5 h-5 text-[#48cae4] shrink-0" />
                        <span className="text-sm text-[#b9d6f2]">
                          {feature}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </Card>
            </motion.div>

            {/* Feature card: speech recognition */}
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
            >
              <Card className="p-8 bg-linear-to-br from-white/10 to-white/5 backdrop-blur-xl border border-white/20 hover:border-[#52b788]/40 transition-all duration-500 h-full group">
                <div className="absolute inset-0 bg-linear-to-br from-[#52b788]/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-xl" />
                <div className="relative">
                  <div className="w-20 h-20 rounded-2xl bg-linear-to-br from-[#40916c]/30 to-[#52b788]/30 flex items-center justify-center mb-6">
                    <Mic className="w-10 h-10 text-[#52b788]" />
                  </div>
                  <h3 className="text-white text-2xl mb-4">
                    Real-Time Speech Recognition
                  </h3>
                  <p className="text-[#8d99ae] leading-relaxed mb-6">
                    Hands-free operation with streaming speech-to-text powered
                    by advanced voice recognition. Speak your queries naturally
                    and watch them appear in real-time, perfect for field work
                    and multitasking scenarios.
                  </p>
                  <div className="space-y-3">
                    {[
                      "Continuous streaming transcription",
                      "Real-time text updates as you speak",
                      "Browser-based voice recognition",
                      "Hands-free query input",
                    ].map((feature, i) => (
                      <div key={i} className="flex items-center gap-3">
                        <CheckCircle2 className="w-5 h-5 text-[#52b788] shrink-0" />
                        <span className="text-sm text-[#b9d6f2]">
                          {feature}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </Card>
            </motion.div>
          </div>
        </div>
      </div>

      {/* Inline CSS for animated gradient text used in the hero */}
      <style>{`
        @keyframes gradient {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        .animate-gradient {
          background-size: 200% auto;
          animation: gradient 6s linear infinite;
        }
      `}</style>
    </div>
  );
}
