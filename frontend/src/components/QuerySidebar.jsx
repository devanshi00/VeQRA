import { useEffect, useRef, useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import {
  X,
  Play,
  Sparkles,
  MicOff,
  Mic,
  Languages,
  Download,
  Target,
  Binary,
  Hash,
  MessageSquare,
  Space,
} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { motion, AnimatePresence } from "framer-motion";
import SpeechRecognition, {
  useSpeechRecognition,
} from "react-speech-recognition";
import { toast } from "react-toastify";
import axios from "axios";

/**
 * Query types offered in the UI.
 * Each entry contains:
 *  - value: internal identifier
 *  - label: displayed label
 *  - icon: icon component to render
 *  - description: short helper text shown below the selector
 */
const queryTypes = [
  {
    value: "binary",
    label: "Binary",
    icon: Binary,
    description: "Yes/No questions",
  },
  {
    value: "numeric",
    label: "Numeric",
    icon: Hash,
    description: "Quantitative analysis",
  },
  {
    value: "semantic",
    label: "Semantic",
    icon: MessageSquare,
    description: "Descriptive analysis",
  },
];

/**
 * QuerySidebar
 * - Slide-out sidebar that contains:
 *    - Research Query UI (free-text + voice input + query type)
 *    - Object Grounding UI (targeted image grounding requests)
 *    - Downloadable chat report action
 *
 * Props (brief):
 *  - isOpen, onClose: control sidebar visibility
 *  - query, onQueryChange, onRunQuery, isProcessing: research query flow
 *  - groundingQuery, setGroundingQuery, onRunGrounding, isGroundingProcessing: grounding flow
 *  - onDownloadReport: export chat/report action
 *  - queryType, setQueryType: selected analysis type (binary/numeric/semantic)
 *  - theme: "dark" | "light" (styling toggle)
 */
export function QuerySidebar({
  isOpen,
  onClose,
  query,
  onQueryChange,
  onRunQuery,
  isProcessing,
  onDownloadReport,
  queryType,
  setQueryType,
  theme,
  groundingQuery,
  setGroundingQuery,
  onRunGrounding,
  isGroundingProcessing = false,
}) {
  // ref to the sidebar DOM element (kept for possible future focus/animation needs)
  const sidebarRef = useRef(null);
  const isDark = theme === "dark";

  // local state tracking which input mode is using voice (research|grounding|null)
  const [voiceInputMode, setVoiceInputMode] = useState(null);

  // Speech recognition hooks provide transcript, listening state and helpers
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition,
  } = useSpeechRecognition();

  /**
   * Keep the current transcript in sync with the appropriate input field.
   * - If voiceInputMode === "research", update the research query input.
   * - If voiceInputMode === "grounding", update the grounding input.
   */
  useEffect(() => {
    if (transcript) {
      if (voiceInputMode === "research") {
        onQueryChange(transcript);
      } else if (voiceInputMode === "grounding") {
        setGroundingQuery(transcript);
      }
    }
    // transcript and voiceInputMode drives this effect
  }, [transcript, voiceInputMode, onQueryChange]);

  // Voice control handlers -------------------------------------------------

  // Start continuous listening and route transcript to research query
  const handleStartListeningResearch = () => {
    resetTranscript();
    onQueryChange("");
    setVoiceInputMode("research");
    SpeechRecognition.startListening({
      continuous: true,
      language: "en-US",
    });
  };

  // Start continuous listening and route transcript to grounding input
  const handleStartListeningGrounding = () => {
    resetTranscript();
    setGroundingQuery("");
    setVoiceInputMode("grounding");
    SpeechRecognition.startListening({
      continuous: true,
      language: "en-US",
    });
  };

  // Stop speech recognition and clear voice input mode
  const handleStopListening = () => {
    SpeechRecognition.stopListening();
    setVoiceInputMode(null);
  };

  /**
   * handleRunResearchQuery
   * - Call onRunQuery to execute the analysis.
   */
  const handleRunResearchQuery = async () => {
    onRunQuery();
  };

  // Render -----------------------------------------------------------------

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            ref={sidebarRef}
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className={`fixed right-0 top-0 h-full sm:w-[400px] shadow-2xl z-50 overflow-hidden flex flex-col ${
              isDark
                ? "bg-[#1b263b]/95 backdrop-blur-xl border-l border-white/10"
                : "bg-white border-l border-[#e3edf3]"
            }`}
          >
            {/* Header: title, animated accent and close button */}
            <div
              className={`flex items-center justify-between p-6 border-b relative overflow-hidden ${
                isDark
                  ? "border-white/10 bg-white/5"
                  : "border-[#e3edf3] bg-[#f4f7fa]"
              }`}
            >
              <div className="absolute inset-0 opacity-30 pointer-events-none">
                <div
                  className={`absolute top-0 right-0 w-32 h-32 rounded-full blur-3xl ${
                    isDark ? "bg-[#48cae4]/20" : "bg-[#0077b6]/10"
                  }`}
                />
              </div>
              <div className="relative z-10">
                <div className="flex items-center gap-2 mb-1">
                  {/* Small rotating sparkle to add motion to header */}
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{
                      duration: 3,
                      repeat: Infinity,
                      ease: "linear",
                    }}
                  >
                    <Sparkles
                      className={`w-4 h-4 ${
                        isDark ? "text-[#48cae4]" : "text-[#0077b6]"
                      }`}
                    />
                  </motion.div>
                  <h2
                    className={`font-semibold ${
                      isDark ? "text-white" : "text-[#1b263b]"
                    }`}
                  >
                    Query Sidebar
                  </h2>
                </div>
                <p
                  className={`text-xs ${
                    isDark ? "text-[#8d99ae]" : "text-[#43515f]"
                  }`}
                >
                  Explore satellite data insights
                </p>
              </div>

              {/* Close control */}
              <motion.div
                whileHover={{ scale: 1.1, rotate: 90 }}
                whileTap={{ scale: 0.9 }}
              >
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onClose}
                  className={`rounded-lg transition-all relative z-10 ${
                    isDark ? "hover:bg-white/10 text-white" : "hover:bg-[#e3edf3]"
                  }`}
                >
                  <X className="w-5 h-5" />
                </Button>
              </motion.div>
            </div>

            {/* Main scrollable content */}
            <div className="flex-1 overflow-y-auto p-6 h-full space-y-6">
              <div className="space-y-4">
                {/* Research Query section header */}
                <div className="flex items-center gap-2">
                  <div
                    className={`w-1 h-6 rounded-full ${
                      isDark ? "bg-[#48cae4]" : "bg-[#0077b6]"
                    }`}
                  />
                  <h3
                    className={`font-semibold ${
                      isDark ? "text-white" : "text-[#1b263b]"
                    }`}
                  >
                    Research Query
                  </h3>
                </div>

                {/* Query type selector, small buttons for binary/numeric/semantic */}
                <div className="space-y-3">
                  <Label
                    htmlFor="query-type"
                    className={`text-sm ${
                      isDark ? "text-[#8d99ae]" : "text-[#43515f]"
                    }`}
                  >
                    Query Type
                  </Label>
                  <div className="grid grid-cols-3 gap-2">
                    {queryTypes.map((type) => {
                      const Icon = type.icon;
                      const isSelected = queryType === type.value;
                      return (
                        <motion.button
                          key={type.value}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          onClick={() => setQueryType(type.value)}
                          className={`p-3 rounded-lg border transition-all ${
                            isSelected
                              ? isDark
                                ? "bg-[#48cae4]/20 border-[#48cae4] text-white"
                                : "bg-[#0077b6]/10 border-[#0077b6] text-[#0077b6]"
                              : isDark
                              ? "bg-white/5 border-white/10 text-[#8d99ae] hover:bg-white/10"
                              : "bg-[#f4f7fa] border-[#c7d4de] text-[#43515f] hover:bg-white"
                          }`}
                        >
                          <Icon className="w-4 h-4 mx-auto mb-1" />
                          <div className="text-xs font-medium">
                            {type.label}
                          </div>
                        </motion.button>
                      );
                    })}
                  </div>
                  <p
                    className={`text-xs ${
                      isDark ? "text-[#8d99ae]" : "text-[#43515f]"
                    }`}
                  >
                    {queryTypes.find((t) => t.value === queryType)?.description}
                  </p>
                </div>

                {/* Text input for the research query with optional voice button */}
                <div className="space-y-3">
                  <Label
                    htmlFor="query-input"
                    className={`text-sm ${
                      isDark ? "text-[#8d99ae]" : "text-[#43515f]"
                    }`}
                  >
                    Your Question
                  </Label>
                  <div className="flex gap-2">
                    <div className="flex-1 relative">
                      <Input
                        id="query-input"
                        value={query}
                        onChange={(e) => onQueryChange(e.target.value)}
                        placeholder="e.g., Are there water bodies in this region?"
                        className={`pr-12 transition-all ${
                          isDark
                            ? "bg-white/5 border-white/10 text-white placeholder:text-[#8d99ae] focus:bg-white/10 focus:border-[#48cae4]"
                            : "bg-[#f4f7fa] border-[#c7d4de] text-[#1b263b] placeholder:text-[#8d99ae] focus:bg-white focus:border-[#0077b6]"
                        }`}
                      />

                      {/* Voice input toggle (only shown if browser supports it) */}
                      {browserSupportsSpeechRecognition && (
                        <motion.div
                          className="absolute right-2 top-1/2 -translate-y-1/2"
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                        >
                          <Button
                            type="button"
                            size="icon"
                            variant="ghost"
                            onClick={
                              listening && voiceInputMode === 'research'
                                ? handleStopListening
                                : handleStartListeningResearch
                            }
                            className={`h-8 w-8 rounded-lg transition-all ${
                              listening
                                ? isDark
                                  ? "bg-red-500/20 hover:bg-red-500/30 text-red-400"
                                  : "bg-red-500/10 hover:bg-red-500/20 text-red-600"
                                : isDark
                                ? "hover:bg-white/10 text-[#48cae4]"
                                : "hover:bg-[#e3edf3] text-[#0077b6]"
                            }`}
                          >
                            {/* Change icon to indicate active listening */}
                            {listening ? (
                              <motion.div
                                animate={{ scale: [1, 1.2, 1] }}
                                transition={{ duration: 1, repeat: Infinity }}
                              >
                                <MicOff className="w-4 h-4" />
                              </motion.div>
                            ) : (
                              <Mic className="w-4 h-4" />
                            )}
                          </Button>
                        </motion.div>
                      )}
                    </div>

                    {/* Run query button (uses Play icon and shows spinner animation when processing) */}
                    <motion.div
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <Button
                        onClick={handleRunResearchQuery}
                        disabled={!query.trim() || isProcessing || isGroundingProcessing}
                        className={`px-6 rounded-lg shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50 ${
                          isDark
                            ? "bg-linear-to-r from-[#48cae4] to-[#00b4d8] hover:from-[#48cae4] hover:to-[#0096c7] text-[#0d1b2a]"
                            : "bg-linear-to-r from-[#0077b6] to-[#005f8f] hover:from-[#0077b6] hover:to-[#0099cc] text-white"
                        }`}
                      >
                        {isProcessing ? (
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{
                              duration: 1,
                              repeat: Infinity,
                              ease: "linear",
                            }}
                          >
                            <Play className="w-4 h-4" />
                          </motion.div>
                        ) : (
                          <Play className="w-4 h-4" />
                        )}
                      </Button>
                    </motion.div>
                  </div>

                  {/* Small listening indicator shown when voice input is active */}
                  {listening && voiceInputMode === "research" && (
                    <p
                      className={`text-xs flex items-center gap-2 ${
                        isDark ? "text-[#8d99ae]" : "text-[#43515f]"
                      }`}
                    >
                      <motion.span
                        animate={{ opacity: [0.5, 1, 0.5] }}
                        transition={{ duration: 1, repeat: Infinity }}
                        className="inline-block w-2 h-2 rounded-full bg-red-500"
                      />
                      Listening...
                    </p>
                  )}
                </div>
              </div>

              {/* Divider */}
              <div
                className={`h-px ${isDark ? "bg-white/10" : "bg-[#e3edf3]"}`}
              />

              {/* Object Grounding section */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <div
                    className={`w-1 h-6 rounded-full ${
                      isDark ? "bg-[#48cae4]" : "bg-[#0077b6]"
                    }`}
                  />
                  <h3
                    className={`font-semibold ${
                      isDark ? "text-white" : "text-[#1b263b]"
                    }`}
                  >
                    Object Grounding
                  </h3>
                </div>

                {/* Grounding text input + voice + run button */}
                <div className="space-y-3">
                  <Label
                    htmlFor="grounding-input"
                    className={`text-sm ${
                      isDark ? "text-[#8d99ae]" : "text-[#43515f]"
                    }`}
                  >
                    Grounding Query
                  </Label>
                  <div className="flex gap-2">
                    <div className="flex-1 relative">
                      <Input
                        id="grounding-input"
                        value={groundingQuery}
                        onChange={(e) => setGroundingQuery(e.target.value)}
                        placeholder="e.g., Highlight all buildings and roads"
                        className={`pr-12 transition-all ${
                          isDark
                            ? "bg-white/5 border-white/10 text-white placeholder:text-[#8d99ae] focus:bg-white/10 focus:border-[#48cae4]"
                            : "bg-[#f4f7fa] border-[#c7d4de] text-[#1b263b] placeholder:text-[#8d99ae] focus:bg-white focus:border-[#0077b6]"
                        }`}
                      />

                      {/* Voice control for grounding input */}
                      {browserSupportsSpeechRecognition && (
                        <motion.div
                          className="absolute right-2 top-1/2 -translate-y-1/2"
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                        >
                          <Button
                            type="button"
                            size="icon"
                            variant="ghost"
                            onClick={
                              listening && voiceInputMode === "grounding"
                                ? handleStopListening
                                : handleStartListeningGrounding
                            }
                            className={`h-8 w-8 rounded-lg transition-all ${
                              listening && voiceInputMode === "grounding"
                                ? isDark
                                  ? "bg-red-500/20 hover:bg-red-500/30 text-red-400"
                                  : "bg-red-500/10 hover:bg-red-500/20 text-red-600"
                                : isDark
                                ? "hover:bg-white/10 text-[#48cae4]"
                                : "hover:bg-[#e3edf3] text-[#0077b6]"
                            }`}
                          >
                            {listening && voiceInputMode === "grounding" ? (
                              <motion.div
                                animate={{ scale: [1, 1.2, 1] }}
                                transition={{ duration: 1, repeat: Infinity }}
                              >
                                <MicOff className="w-4 h-4" />
                              </motion.div>
                            ) : (
                              <Mic className="w-4 h-4" />
                            )}
                          </Button>
                        </motion.div>
                      )}
                    </div>

                    <motion.div
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <Button
                        onClick={() => onRunGrounding(groundingQuery)}
                        disabled={!groundingQuery.trim() || isGroundingProcessing || isProcessing}
                        className={`px-6 rounded-lg shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50 ${
                          isDark
                            ? "bg-linear-to-r from-[#48cae4] to-[#00b4d8] hover:from-[#48cae4] hover:to-[#0096c7] text-[#0d1b2a]"
                            : "bg-linear-to-r from-[#0077b6] to-[#005f8f] hover:from-[#0077b6] hover:to-[#0099cc] text-white"
                        }`}
                      >
                        {isGroundingProcessing ? (
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{
                              duration: 1,
                              repeat: Infinity,
                              ease: "linear",
                            }}
                          >
                            <Play className="w-4 h-4" />
                          </motion.div>
                        ) : (
                          <Play className="w-4 h-4" />
                        )}
                      </Button>
                    </motion.div>
                  </div>

                  {/* Listening indicator for grounding voice mode */}
                  {listening && voiceInputMode === "grounding" && (
                    <p
                      className={`text-xs flex items-center gap-2 ${
                        isDark ? "text-[#8d99ae]" : "text-[#43515f]"
                      }`}
                    >
                      <motion.span
                        animate={{ opacity: [0.5, 1, 0.5] }}
                        transition={{ duration: 1, repeat: Infinity }}
                        className="inline-block w-2 h-2 rounded-full bg-red-500"
                      />
                      Listening...
                    </p>
                  )}
                </div>
              </div>

              {/* Divider */}
              <div
                className={`h-px ${isDark ? "bg-white/10" : "bg-[#e3edf3]"}`}
              />

              {/* Chat report download area */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <div
                    className={`w-1 h-6 rounded-full ${
                      isDark ? "bg-[#48cae4]" : "bg-[#0077b6]"
                    }`}
                  />
                  <h3
                    className={`font-semibold ${
                      isDark ? "text-white" : "text-[#1b263b]"
                    }`}
                  >
                    Chat Report
                  </h3>
                </div>
                <motion.div
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <Button
                    onClick={onDownloadReport}
                    className={`w-full rounded-lg shadow-lg hover:shadow-xl transition-all duration-300 ${
                      isDark
                        ? "bg-linear-to-r from-[#48cae4] to-[#00b4d8] hover:from-[#48cae4] hover:to-[#0096c7] text-[#0d1b2a]"
                        : "bg-linear-to-r from-[#0077b6] to-[#005f8f] hover:from-[#0077b6] hover:to-[#0099cc] text-white"
                    }`}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download Report
                  </Button>
                </motion.div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
