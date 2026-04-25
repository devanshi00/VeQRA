import { useRef, useEffect } from "react";
import { motion, useScroll, useTransform } from "motion/react";
import gsap from "gsap";

/**
 * BackgroundEffects
 * - Renders a subtle animated grid, a parallax image block and floating particles.
 * - isDark: boolean that toggles color/opacity choices for dark/light themes.
 */
export function BackgroundEffects({ isDark }) {
  // refs to DOM nodes we animate with GSAP
  const gridRef = useRef(null);
  const particlesRef = useRef(null);

  // useScroll + useTransform (from motion) to create a small parallax y offset
  const { scrollY } = useScroll();
  const y1 = useTransform(scrollY, [0, 300], [0, 100]); // map scroll range to y offset

  useEffect(() => {
    // animate the grid background position infinitely to create a drifting grid
    if (gridRef.current) {
      gsap.to(gridRef.current, {
        backgroundPosition: "100px 100px",
        duration: 15,
        repeat: -1,
        ease: "none",
      });
    }

    // animate each particle with slight random motion and staggered delays
    if (particlesRef.current) {
      const particles = particlesRef.current.querySelectorAll(".bg-particle");
      particles.forEach((particle, i) => {
        gsap.to(particle, {
          y: -50 - Math.random() * 100, // float upward by a random amount
          x: Math.random() * 50 - 25,   // small horizontal drift
          opacity: 0.3,
          duration: 4 + Math.random() * 3, // varied duration per particle
          repeat: -1,
          delay: i * 0.3, // stagger start times
          ease: "power1.out",
        });
      });
    }
    // empty deps: run once on mount
  }, []);

  return (
    <>
      {/* Animated grid overlay — thin grid lines via background-image */}
      <div
        ref={gridRef}
        className={`fixed inset-0 pointer-events-none ${
          isDark ? "opacity-5" : "opacity-[0.02]"
        }`}
        style={{
          backgroundImage: `
            linear-gradient(${
              isDark ? "rgba(72, 202, 228, 0.3)" : "rgba(27, 38, 59, 0.1)"
            } 1px, transparent 1px),
            linear-gradient(90deg, ${
              isDark ? "rgba(72, 202, 228, 0.3)" : "rgba(27, 38, 59, 0.1)"
            } 1px, transparent 1px)
          `,
          backgroundSize: "50px 50px",
        }}
      />

      {/* Parallax image block — moves slightly with scroll (motion.div using y transform) */}
      <motion.div
        style={{ y: y1 }}
        className={`fixed top-0 right-0 w-[500px] h-[500px] pointer-events-none ${
          isDark ? "opacity-10" : "opacity-5"
        } blur-sm`}
      >
        {/* Gradient overlay to blend image with background; switches by theme */}
        <div
          className={`absolute inset-0 ${
            isDark
              ? "bg-linear-to-l from-transparent to-[#0a1929]"
              : "bg-linear-to-l from-transparent to-[#e3edf3]"
          }`}
        />
        {/* Decorative image — purely visual */}
        <img
          src="https://images.unsplash.com/photo-1640796433065-f423a9d9a5fd?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxvcmJpdGFsJTIwc2F0ZWxsaXRlJTIwdmlld3xlbnwxfHx8fDE3NjI3OTIyMDN8MA&ixlib=rb-4.1.0&q=80&w=1080"
          alt=""
          className="w-full h-full object-cover"
        />
      </motion.div>

      {/* Floating particles container — small circular elements animated by GSAP */}
      <div
        ref={particlesRef}
        className="fixed inset-0 overflow-hidden pointer-events-none"
      >
        {[...Array(15)].map((_, i) => (
          <div
            key={i}
            className={`bg-particle absolute w-1 h-1 rounded-full ${
              isDark ? "bg-[#48cae4]" : "bg-[#0077b6]"
            }`}
            style={{
              left: `${Math.random() * 100}%`,  // random horizontal start
              top: `${80 + Math.random() * 20}%`, // start near bottom for upward float
              opacity: 0.4,
            }}
          />
        ))}
      </div>
    </>
  );
}
