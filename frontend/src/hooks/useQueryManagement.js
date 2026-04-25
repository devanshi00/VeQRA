import axios from "axios";
import { useState } from "react";
import { backendLink } from "../../../config";
import { toast } from "react-toastify";

/**
 * useQueryManagement
 * - Encapsulates logic for running queries (VQA) and grounding requests against
 *   the backend for the currently selected chat session.
 * - Manages local UI state: query text, grounding text, processing flags,
 *   accumulated query results, selected voice language and query type.
 *
 * Params:
 *  - currentChatId: id of the currently active chat/session (used when posting queries)
 *  - setChatSessions: setter to update the global chatSessions list (adds results, updates title/lastActivity)
 *  - theme: string "dark" | "light" used to choose toast theme
 *
 * Returns:
 *  - Local state values and handlers for running queries / grounding
 */
export function useQueryManagement(currentChatId, setChatSessions, theme) {
  // Controlled research query input
  const [query, setQuery] = useState("");
  // Controlled grounding query input (object-grounding style requests)
  const [groundingQuery, setGroundingQuery] = useState("");
  // Busy flag for research query requests
  const [isProcessing, setIsProcessing] = useState(false);
  // List of query result objects produced in this session (most-recent-first)
  const [queryResults, setQueryResults] = useState([]);
  // Current selected query type (binary/numeric/semantic)
  const [queryType, setQueryType] = useState("semantic");
  // Busy flag for grounding requests
  const [isGroundingProcessing, setIsGroundingProcessing] = useState(false);
  // Theme boolean derived from the provided theme string (used for toast styling)
  const isDark = theme === "dark";

  /**
   * handleRunQuery
   * - Validates input, posts a VQA query to backend and updates both local queryResults
   *   and the corresponding chat entry in the global chatSessions list.
   * - On success prepends the new result so newest results appear first.
   */
  const handleRunQuery = async () => {
    if (!query.trim()) return;

    setIsProcessing(true);
    try {
      const res = await axios.post(`${backendLink}/api/query/vqa`, {
        chatId: currentChatId,
        query,
        queryType
      });

      // Normalize backend response to UI-friendly result object
      const result = {
        id: res.data.id,
        query,
        textAnswer: res.data.text_answer,
        generatedImage: res.data.generated_image,
        timestamp: new Date(res.data.created_at),
      };

      // Add to local results list (prepended)
      setQueryResults((prev) => [result, ...prev]);

      // If attached to a chat, also update the global sessions list
      if (currentChatId) {
        setChatSessions((prev) =>
          prev.map((chat) =>
            chat.id === currentChatId
              ? {
                  ...chat,
                  queryResults: [result, ...chat.queryResults],
                  lastActivity: new Date(),
                  // If chat had no title (first query), use the query as title
                  title: chat.queryResults.length === 0 ? query : chat.title,
                }
              : chat
          )
        );
      }
    } catch (error) {
      console.log(error);
      toast.error("Error running the query!", {
        position: "top-center",
        autoClose: 3000,
        theme: isDark ? "dark" : "light",
      });
    }

    setIsProcessing(false);
    // clear the input after submission
    setQuery("");
  };

  /**
   * handleRunGrounding
   * - Similar to handleRunQuery but posts to the grounding endpoint.
   * - Adds returned grounding result to local/global lists and clears grounding input.
   */
  const handleRunGrounding = async (query) => {
    setIsGroundingProcessing(true);

    try {
      const res = await axios.post(`${backendLink}/api/query/grounding`, {
        chatId: currentChatId,
        query: query,
      });

      const result = {
        id: res.data.id,
        query,
        textAnswer: res.data.text_answer,
        generatedImage: res.data.generated_image,
        timestamp: new Date(res.data.created_at),
      };

      // Prepend to local results and clear grounding input
      setQueryResults((prev) => [result, ...prev]);
      setGroundingQuery("");

      // Mirror update to the global chat sessions list
      if (currentChatId) {
        setChatSessions((prev) =>
          prev.map((chat) =>
            chat.id === currentChatId
              ? {
                  ...chat,
                  queryResults: [result, ...chat.queryResults],
                  lastActivity: new Date(),
                  title: chat.queryResults.length === 0 ? query : chat.title,
                }
              : chat
          )
        );
      }
    } catch (error) {
      console.log(error);
      toast.error("Error running the query!", {
        position: "top-center",
        autoClose: 3000,
        theme: isDark ? "dark" : "light",
      });
    }

    setIsGroundingProcessing(false);
  };

  // Expose state and handlers to components
  return {
    query,
    setQuery,
    isProcessing,
    queryResults,
    setQueryResults,
    handleRunQuery,
    handleRunGrounding,
    groundingQuery,
    setGroundingQuery,
    isGroundingProcessing,
    setIsGroundingProcessing,
    queryType,
    setQueryType,
  };
}
