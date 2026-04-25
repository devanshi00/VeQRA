import { motion, AnimatePresence } from "motion/react";
import { Card } from "./ui/card";
import { ImageWithFallback } from "./ImageWithFallback";
import { MapPin, Sparkles, Calendar } from "lucide-react";

/**
 * QueryResultsList
 * - Renders a list of past query results (text answers and optional generated images).
 * - Props:
 *   - isDark: boolean to toggle dark/light styling
 *   - queryResults: array of result objects:
 *       { id, query, textAnswer, generatedImage, timestamp }
 */
export function QueryResultsList({ isDark, queryResults }) {
  // Nothing to render when there are no results
  if (queryResults.length === 0) return null;

  return (
    // Animate presence so the whole block fades in/out with mounting/unmounting
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="space-y-4"
      >
        {/* Section header with icon + title */}
        <div className="flex items-center gap-2">
          <MapPin
            className={`w-5 h-5 ${
              isDark ? "text-[#48cae4]" : "text-[#0077b6]"
            }`}
          />
          <h2 className={isDark ? "text-white" : "text-[#1b263b]"}>
            Analysis Results
          </h2>
          <Sparkles
            className={`w-4 h-4 ${
              isDark ? "text-[#48cae4]" : "text-[#0077b6]"
            }`}
          />
        </div>

        {/* List of result cards */}
        <div className="grid gap-4">
          {queryResults.map((result, index) => (
            // Each card animates in with a small stagger based on index
            <motion.div
              key={result.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
              whileHover={{ y: -4 }}
            >
              <Card
                className={`overflow-hidden rounded-2xl border transition-all ${
                  isDark
                    ? "bg-white/5 border-white/10 backdrop-blur-xl hover:bg-white/10"
                    : "bg-white border-[#e3edf3] shadow-md hover:shadow-xl"
                }`}
              >
                {/* Header row: shows the query label and timestamp */}
                <div
                  className={`p-4 border-b ${
                    isDark
                      ? "bg-white/5 border-white/10"
                      : "bg-[#f4f7fa] border-[#e3edf3]"
                  }`}
                >
                  <div className="flex flex-col sm:flex-row items-center justify-between">
                    <p
                      className={`text-sm italic self-start ${
                        isDark ? "text-[#8d99ae]" : "text-[#43515f]"
                      }`}
                    >
                      "Query"
                    </p>
                    <span
                      className={`text-xs self-end flex items-center gap-1 ${
                        isDark ? "text-[#8d99ae]" : "text-[#43515f]"
                      }`}
                    >
                      <Calendar className="w-3 h-3" />
                      {/* Friendly time display from the timestamp */}
                      {result.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                </div>

                {/* Body: query text, optional text answer and optional generated image */}
                <div className="px-3 pb-3 space-y-4">
                  {/* User's query (primary content) */}
                  <p className={`${isDark ? "text-white" : "text-[#1b263b]"} font-bold`}>{result.query}</p>

                  {/* Text answer block (if present) */}
                  {result.textAnswer ? (
                    <div
                      className={`p-4 rounded-lg border ${
                        isDark
                          ? "bg-white/5 border-white/10"
                          : "bg-[#e3edf3] border-[#c7d4de]"
                      }`}
                    >
                      <p className={isDark ? "text-white" : "text-[#1b263b]"}>
                        {result.textAnswer}
                      </p>
                    </div>
                  ) : null}

                  {/* Generated analysis image (if present) */}
                  {result.generatedImage && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.2 }}
                    >
                      <h4
                        className={`text-sm mb-2 ${
                          isDark ? "text-[#8d99ae]" : "text-[#43515f]"
                        }`}
                      >
                        Generated Analysis Image
                      </h4>

                      {/* Image container uses ImageWithFallback which handles load errors and preview */}
                      <div
                        className={`relative rounded-lg overflow-hidden border ${
                          isDark
                            ? "bg-[#2d3e50] border-white/10"
                            : "bg-[#e8eff4] border-[#e3edf3]"
                        }`}
                      >
                        <ImageWithFallback
                          src={result.generatedImage}
                          alt="Generated Analysis"
                          className="w-full h-full object-cover"
                        />
                      </div>
                    </motion.div>
                  )}
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
