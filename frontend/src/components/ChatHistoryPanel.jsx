import { useEffect, useRef } from "react";
import { ScrollArea } from "./ui/scroll-area";
import { Button } from "./ui/button";
import { motion, AnimatePresence } from "motion/react";
import {
  MessageSquare,
  Clock,
  X,
  Menu,
  Plus,
  Trash2,
  Image as ImageIcon,
  Sparkles,
} from "lucide-react";
import gsap from "gsap";

/**
 * ChatHistoryPanel
 * - Displays a slide-over panel listing past analysis sessions.
 * - Props:
 *   - chatSessions: array of session objects
 *   - currentChatId: id of the currently selected session
 *   - onSelectChat: callback when a session is selected
 *   - onNewChat: callback to create a new session
 *   - onDeleteChat: callback to delete a session
 *   - isOpen: boolean controlling panel visibility
 *   - onToggle: toggle callback to open/close the panel
 *   - theme: "dark" | "light" (controls styling)
 */
export function ChatHistoryPanel({
  chatSessions,
  currentChatId,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  isOpen,
  onToggle,
  theme,
}) {
  // ref for the slide-over panel DOM node (used for GSAP entrance animation)
  const panelRef = useRef(null);
  const isDark = theme === "dark";

  useEffect(() => {
    // When the panel opens, animate each chat item in with a small staggered slide
    if (isOpen && panelRef.current) {
      const items = panelRef.current.querySelectorAll(".chat-item");
      gsap.fromTo(
        items,
        { x: -30, opacity: 0 },
        {
          x: 0,
          opacity: 1,
          duration: 0.3,
          stagger: 0.05,
          ease: "power2.out",
        }
      );
    }
    // Re-run animation when isOpen or chatSessions change
  }, [isOpen, chatSessions]);

  // Helper: format a Date object into a friendly label
  const formatDate = (date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) return "Today";
    if (days === 1) return "Yesterday";
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString();
  };

  // Helper: group chat sessions into buckets by recency for UI sections
  const groupChatsByDate = () => {
    const groups = {
      Today: [],
      Yesterday: [],
      "Previous 7 Days": [],
      Older: [],
    };

    chatSessions.forEach((chat) => {
      const now = new Date();
      const diff = now.getTime() - chat.lastActivity.getTime();
      const days = Math.floor(diff / (1000 * 60 * 60 * 24));

      if (days === 0) groups.Today.push(chat);
      else if (days === 1) groups.Yesterday.push(chat);
      else if (days < 7) groups["Previous 7 Days"].push(chat);
      else groups.Older.push(chat);
    });

    return groups;
  };

  return (
    <>
      {/* Backdrop: a blurred overlay that closes the panel when clicked */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onToggle}
            className={`fixed inset-0 z-40 backdrop-blur-sm ${
              isDark ? "bg-black/50" : "bg-black/20"
            }`}
          />
        )}
      </AnimatePresence>

      {/* Slide-over panel itself */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            ref={panelRef}
            initial={{ x: "-100%" }}
            animate={{ x: 0 }}
            exit={{ x: "-100%" }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
            className={`fixed left-0 top-0 h-full w-80 shadow-2xl z-50 flex flex-col ${
              isDark
                ? "bg-[#1b263b]/95 backdrop-blur-xl border-r border-white/10"
                : "bg-white border-r border-[#e3edf3]"
            }`}
          >
            {/* Header: title, icon and close button */}
            <div
              className={`flex items-center justify-between p-4 border-b ${
                isDark
                  ? "border-white/10 bg-white/5"
                  : "border-[#e3edf3] bg-[#f4f7fa]"
              }`}
            >
              <div className="flex items-center gap-2">
                {/* Animated icon to add visual polish */}
                <motion.div
                  animate={{ rotate: [0, 10, -10, 0] }}
                  transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
                >
                  <MessageSquare
                    className={`w-5 h-5 ${
                      isDark ? "text-[#48cae4]" : "text-[#0077b6]"
                    }`}
                  />
                </motion.div>
                <h2 className={isDark ? "text-white" : "text-[#1b263b]"}>
                  Analysis Sessions
                </h2>
              </div>

              {/* Close button (also animated on hover/tap) */}
              <motion.div
                whileHover={{ scale: 1.1, rotate: 90 }}
                whileTap={{ scale: 0.9 }}
              >
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onToggle}
                  className={`rounded-lg transition-all ${
                    isDark ? "hover:bg-white/10 text-white" : "hover:bg-[#e3edf3]"
                  }`}
                >
                  <X className="w-5 h-5" />
                </Button>
              </motion.div>
            </div>

            {/* New session button area */}
            <div
              className={`p-4 border-b ${
                isDark ? "border-white/10" : "border-[#e3edf3]"
              }`}
            >
              <motion.div
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Button
                  onClick={() => {
                    // create a new chat and close the panel
                    onNewChat();
                    onToggle();
                  }}
                  className={`w-full rounded-xl shadow-lg transition-all duration-300 relative overflow-hidden group ${
                    isDark
                      ? "bg-linear-to-r from-[#48cae4] to-[#00b4d8] hover:from-[#48cae4] hover:to-[#0096c7] text-[#0d1b2a]"
                      : "bg-linear-to-r from-[#0077b6] to-[#005f8f] hover:from-[#0077b6] hover:to-[#0099cc] text-white"
                  }`}
                >
                  {/* Subtle animated sheen to the button */}
                  <div className="absolute inset-0 bg-linear-to-r from-white/0 via-white/20 to-white/0 translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-1000" />
                  <span className="relative flex items-center justify-center gap-2">
                    <Plus className="w-4 h-4" />
                    New Analysis Session
                  </span>
                </Button>
              </motion.div>
            </div>

            {/* Scrollable list area for chat sessions */}
            <ScrollArea className="flex-1 p-4 overflow-y-auto">
              {chatSessions.length === 0 ? (
                // Empty state: prompt to start a new session
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="flex flex-col items-center justify-center py-12 text-center"
                >
                  <motion.div
                    animate={{ y: [-5, 5, -5] }}
                    transition={{
                      duration: 3,
                      repeat: Infinity,
                      ease: "easeInOut",
                    }}
                    className={`w-20 h-20 rounded-2xl flex items-center justify-center mb-4 ${
                      isDark
                        ? "bg-white/5 border border-white/10"
                        : "bg-[#e3edf3] border border-[#c7d4de]"
                    }`}
                  >
                    <MessageSquare
                      className={`w-10 h-10 ${
                        isDark ? "text-[#48cae4]" : "text-[#0077b6]"
                      }`}
                    />
                  </motion.div>
                  <p
                    className={`text-sm ${
                      isDark ? "text-[#8d99ae]" : "text-[#43515f]"
                    }`}
                  >
                    No sessions yet
                  </p>
                  <p
                    className={`text-xs mt-2 ${
                      isDark ? "text-[#8d99ae]" : "text-[#8d99ae]"
                    }`}
                  >
                    Start by uploading an image
                  </p>
                </motion.div>
              ) : (
                // Render grouped chat sections (Today, Yesterday, etc.)
                <div className="space-y-6">
                  {Object.entries(groupChatsByDate()).map(([group, chats]) => {
                    if (chats.length === 0) return null;
                    return (
                      <div key={group} className="space-y-2">
                        {/* Group header with label and small icon */}
                        <div className="flex items-center gap-2 px-2">
                          <h3
                            className={`text-xs ${
                              isDark ? "text-[#8d99ae]" : "text-[#8d99ae]"
                            }`}
                          >
                            {group}
                          </h3>
                          <Sparkles
                            className={`w-3 h-3 ${
                              isDark ? "text-[#48cae4]" : "text-[#0077b6]"
                            }`}
                          />
                        </div>

                        {/* Each chat item */}
                        <div className="space-y-2">
                          {chats.map((chat) => (
                            <motion.div
                              key={chat.id}
                              className="chat-item group relative"
                              whileHover={{ y: 4 }}
                            >
                              {/* Selecting a chat closes the panel and notifies parent */}
                              <button
                                onClick={() => {
                                  onSelectChat(chat.id);
                                  onToggle();
                                }}
                                className={`w-full text-left p-3 rounded-xl border transition-all duration-300 ${
                                  currentChatId === chat.id
                                    ? isDark
                                      ? "bg-white/10 border-[#48cae4] shadow-lg shadow-[#48cae4]/20"
                                      : "bg-[#e3edf3] border-[#0077b6] shadow-md"
                                    : isDark
                                    ? "bg-white/5 border-white/10 hover:bg-white/10 hover:border-[#48cae4]/50"
                                    : "bg-[#f4f7fa] border-[#e3edf3] hover:bg-white hover:border-[#0077b6]/50"
                                }`}
                              >
                                <div className="flex items-start gap-3 pr-8">
                                  {/* Thumbnail: either uploaded image preview or placeholder icon */}
                                  {chat.uploadedImage ? (
                                    <div
                                      className={`w-10 h-10 rounded-lg overflow-hidden shrink-0 border ${
                                        isDark
                                          ? "border-white/10 bg-white/5"
                                          : "border-[#e3edf3] bg-[#e8eff4]"
                                      }`}
                                    >
                                      <img
                                        src={chat.uploadedImage.url}
                                        alt=""
                                        className="w-full h-full object-cover"
                                      />
                                    </div>
                                  ) : (
                                    <div
                                      className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${
                                        isDark ? "bg-white/5" : "bg-[#e3edf3]"
                                      }`}
                                    >
                                      <ImageIcon
                                        className={`w-5 h-5 ${
                                          isDark ? "text-[#8d99ae]" : "text-[#8d99ae]"
                                        }`}
                                      />
                                    </div>
                                  )}

                                  {/* Title, timestamp and query count */}
                                  <div className="flex-1 min-w-0">
                                    <p
                                      className={`text-sm line-clamp-2 mb-1 ${
                                        isDark ? "text-white" : "text-[#1b263b]"
                                      }`}
                                    >
                                      {chat.title.slice(0, 10) + " ..."}
                                    </p>
                                    <div
                                      className={`flex items-center gap-2 text-xs ${
                                        isDark ? "text-[#8d99ae]" : "text-[#8d99ae]"
                                      }`}
                                    >
                                      <Clock className="w-3 h-3" />
                                      <span>
                                        {formatDate(chat.lastActivity)}
                                      </span>
                                    </div>
                                    {chat.queryResults.length > 0 && (
                                      <p
                                        className={`text-xs mt-1 ${
                                          isDark ? "text-[#48cae4]" : "text-[#0077b6]"
                                        }`}
                                      >
                                        {chat.queryResults.length}{" "}
                                        {chat.queryResults.length === 1 ? "query" : "queries"}
                                      </p>
                                    )}
                                  </div>
                                </div>
                              </button>

                              {/* Delete button (stops propagation so it doesn't select the chat) */}
                              <motion.div
                                initial={{ opacity: 0, scale: 0.8 }}
                                whileHover={{ opacity: 1, scale: 1 }}
                                className="absolute top-2 right-2"
                              >
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    onDeleteChat(chat.id);
                                  }}
                                  className="opacity-0 group-hover:opacity-100 transition-all rounded-lg hover:bg-red-50 dark:hover:bg-red-950/30 hover:text-[#c1121f]"
                                >
                                  <Trash2 className="w-4 h-4" />
                                </Button>
                              </motion.div>
                            </motion.div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </ScrollArea>

            {/* Footer: shows total session count with a subtle indicator */}
            <div
              className={`p-4 border-t ${
                isDark
                  ? "border-white/10 bg-white/5"
                  : "border-[#e3edf3] bg-[#f4f7fa]"
              }`}
            >
              <div className="flex items-center justify-center gap-2">
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className={`w-2 h-2 rounded-full ${
                    isDark ? "bg-[#48cae4]" : "bg-[#0077b6]"
                  }`}
                />
                <p
                  className={`text-xs ${
                    isDark ? "text-[#8d99ae]" : "text-[#8d99ae]"
                  }`}
                >
                  {chatSessions.length} session{chatSessions.length !== 1 ? "s" : ""} total
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
