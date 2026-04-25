import { useState, useEffect } from "react";
import { toast } from "react-toastify";
import axios from "axios";
import { backendLink } from "../../../config";

/**
 * useChatManagement
 * - Custom hook that encapsulates fetching and managing chat sessions for the current user.
 * - Responsibilities:
 *    - Load chat sessions from backend for a given user
 *    - Track the currently selected chat and its caption
 *    - Provide helpers to select, create (reset), delete chats and generate image captions
 *
 * Params:
 *  - userData: current logged-in user object (expects .id)
 *  - isDark: theme flag used for toast theming
 *
 * Returns:
 *  - chatSessions, currentChatId, caption, isGenerating and various handler functions
 */
export function useChatManagement(userData, isDark, setSidebarOpen) {
  // list of chat session summaries shown in the UI
  const [chatSessions, setChatSessions] = useState([]);
  // id of the currently selected chat session
  const [currentChatId, setCurrentChatId] = useState(null);
  // generated caption for the current chat (optional)
  const [caption, setCaption] = useState(null);
  // whether a caption generation request is in progress
  const [isGenerating, setIsGenerating] = useState(false);

  // Fetch user's chats when userData becomes available or theme changes (theme used only for toast)
  useEffect(() => {
    if (userData) {
      const fetchChats = async () => {
        try {
          const res = await axios.get(
            `${backendLink}/api/chat/user/${userData.id}`
          );

          // Transform backend shape into UI-friendly chat session objects
          const chatHistory = [];
          res.data.map((chat) => {
            const newChat = {
              id: chat.id,
              title: chat.title,
              uploadedImage: { url: chat.image_url, name: chat.title },
              queryResults: [], // lazy-loaded when a chat is selected
              caption: chat.caption,
              timestamp: new Date(chat.created_at),
              lastActivity: new Date(chat.updated_at),
            };
            chatHistory.push(newChat);
          });

          setChatSessions(chatHistory);
        } catch (error) {
          // Notify user on failure and log for debugging
          toast("Error fetching the chats!", {
            position: "top-center",
            autoClose: 3000,
            theme: isDark ? "dark" : "light",
          });
          console.log(error);
        }
      };
      fetchChats();
    }
  }, [userData, isDark]);

  /**
   * handleSelectChat
   * - Loads a chat's detailed query history and updates selected state.
   * - Parameters:
   *    - chatId: id to select
   *    - setUploadedImage: setter from caller to display the chat's image preview
   *    - setQueryResults: setter from caller to display the chat's query results list
   */
  const handleSelectChat = async (
    chatId,
    setUploadedImage,
    setQueryResults
  ) => {
    const chat = chatSessions.find((c) => c.id === chatId);
    if (chat) {
      // update current selection and UI-bound state
      setCurrentChatId(chatId);
      setUploadedImage(chat.uploadedImage);
      setCaption(chat.caption);

      try {
        // fetch message/query history for the chat
        const res = await axios.get(
          `${backendLink}/api/chat/${chatId}/messages`
        );

        // transform backend messages into UI-friendly result objects
        let queries = [];
        res.data.map((query) => {
          const result = {
            id: query.id,
            query: query.query,
            textAnswer: query.text_answer,
            generatedImage: query.generated_image,
            timestamp: new Date(query.created_at),
          };
          queries.push(result);
        });

        // persist query results into the sessions list and push to caller state
        setChatSessions((prev) =>
          prev.map((chat) =>
            chat.id === chatId
              ? {
                  ...chat,
                  queryResults: queries,
                }
              : chat
          )
        );
        setQueryResults(queries);
        setSidebarOpen(true);
      } catch (error) {
        console.log(error);
        toast.error("Error fetching history!", {
          position: "top-center",
          autoClose: 3000,
          theme: isDark ? "dark" : "light",
        });
      }
    }
  };

  /**
   * handleNewChat
   * - Resets the UI to start a new chat/session (clears image, results, caption and current id)
   * - Optionally clears the file input element if a ref is passed.
   */
  const handleNewChat = (setUploadedImage, setQueryResults, fileInputRef) => {
    setUploadedImage(null);
    setQueryResults([]);
    setCaption("");
    setCurrentChatId(null);
    if (fileInputRef?.current) {
      fileInputRef.current.value = "";
    }
  };

  /**
   * handleDeleteChat
   * - Deletes a chat on the backend and removes it from local state.
   * - If the deleted chat was currently selected, resets UI to a new chat state.
   */
  const handleDeleteChat = async (
    chatId,
    setUploadedImage,
    setQueryResults,
    fileInputRef
  ) => {
    try {
      const res = await axios.delete(`${backendLink}/api/chat/${chatId}`);
      // remove from local list
      setChatSessions((prev) => prev.filter((chat) => chat.id !== chatId));
      // if deleted chat was open, clear the UI
      if (currentChatId === chatId) {
        handleNewChat(setUploadedImage, setQueryResults, fileInputRef);
      }
    } catch (error) {
      console.log(error);
      toast.error("Error deleting chat!", {
        position: "top-center",
        autoClose: 3000,
        theme: isDark ? "dark" : "light",
      });
    }
  };

  /**
   * generateCaption
   * - Requests a generated caption for the currently selected chat from the backend.
   * - Updates local caption state and the corresponding chat entry on success.
   */
  const generateCaption = async () => {
    console.log("here");
    setIsGenerating(true);
    try {
      const response = await axios.post(`${backendLink}/api/query/caption`, {
        chatId: currentChatId,
      });

      if (response.status != 200) {
        throw new Error("Failed to generate caption");
      }

      // update caption in hook state and inside the sessions array
      setCaption(response.data.caption);
      setChatSessions((prev) =>
        prev.map((chat) =>
          chat.id === currentChatId
            ? {
                ...chat,
                caption: response.data.caption,
              }
            : chat
        )
      );
    } catch (error) {
      console.error("Error generating caption:", error);
    } finally {
      setIsGenerating(false);
    }
  };

  // Expose state and handlers to callers
  return {
    chatSessions,
    setChatSessions,
    currentChatId,
    caption,
    isGenerating,
    setCurrentChatId,
    handleSelectChat,
    handleNewChat,
    handleDeleteChat,
    generateCaption,
  };
}
