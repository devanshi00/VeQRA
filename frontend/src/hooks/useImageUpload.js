import { useState, useRef } from "react";
import { toast } from "react-toastify";
import axios from "axios";
import { backendLink } from "../../../config";

/**
 * useImageUpload
 * - Hook that centralizes image upload logic for the app.
 * - Responsibilities:
 *   - Track uploaded image state and related UI flags (loading, dragging)
 *   - Accept files via file picker, drag-and-drop, or a direct image URL
 *   - Validate file types and convert remote image URLs into File objects
 *   - Create a new chat on successful upload and update parent chat state
 *
 * Params:
 *  - userData: current logged-in user (used when creating a new chat)
 *  - isDark: theme flag used for toast styling
 *  - setChatSessions: setter from parent to prepend the newly created chat
 *  - setCurrentChatId: setter to mark the newly created chat as selected
 *  - setSidebarOpen: setter to open the query sidebar after upload
 *
 * Returns:
 *  - uploadedImage, imageLink, isLoading, isDragging, fileInputRef
 *  - handlers: handleFileUpload, handleImageLinkSubmit, drag/drop handlers, clearImage
 */
export function useImageUpload(
  userData,
  isDark,
  setChatSessions,
  setCurrentChatId,
  setSidebarOpen,
) {
  // currently selected/uploaded image object: { url, name }
  const [uploadedImage, setUploadedImage] = useState(null);
  // controlled input state for the Link tab
  const [imageLink, setImageLink] = useState("");
  // UI loading state while converting/link-fetching/uploading
  const [isLoading, setIsLoading] = useState(false);
  // UI drag state used to highlight the drop target
  const [isDragging, setIsDragging] = useState(false);
  // ref to hidden file input so callers can trigger the native picker
  const fileInputRef = useRef(null);

  /**
   * handleFileUpload
   * - Primary file handler used by the native picker and by the link-to-file flow.
   * - Validates mime type, uploads the file to backend, creates a new chat record,
   *   updates parent chat list and opens the sidebar to show the new session.
   *
   * Accepts an event-like object shaped as { target: { files: [File] } } so it
   * can be called directly from input change handlers or programmatically.
   */
  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    setIsLoading(true);
    if (file) {
      // whitelist common image mime types
      const allowedTypes = [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/svg+xml",
      ];

      // basic client-side validation: show toast and clear input if invalid
      if (!allowedTypes.includes(file.type)) {
        toast.error("Invalid file type!", {
          position: "top-center",
          autoClose: 3000,
          theme: isDark ? "dark" : "light",
        });

        // reset the native input value to allow re-selecting same file later
        e.target.value = "";
        setIsLoading(false);
        return;
      }

      // prepare multipart form data and POST to the upload endpoint
      const formData = new FormData();
      formData.append("image", file);

      try {
        const res = await axios.post(`${backendLink}/api/upload`, formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });

        // build a normalized uploaded image object for UI use
        const uploaded = {
          url: res.data.imageUrl,
          name: file.name,
        };

        // Create a new chat/session on the backend associated with this image
        const newChatres = await axios.post(`${backendLink}/api/chat/new`, {
          userId: userData?.id,
          imageUrl: uploaded.url,
          title: file.name,
        });
        
        // Compose a lightweight chat session object for the sessions list
        const newChat = {
          id: newChatres.data.id,
          title: file.name,
          uploadedImage: uploaded,
          queryResults: [],
          timestamp: new Date(),
          lastActivity: new Date(),
        };
        
        // Prepend the new chat to the existing list and set it as the current selection
        setChatSessions((prev) => [newChat, ...prev]);
        setCurrentChatId(newChat.id);
        // store locally so preview components can consume it
        setUploadedImage(uploaded);
        // open the query sidebar so the user can immediately start queries
        setSidebarOpen(true);
      } catch (err) {
        // log and notify user on any upload/create failure
        console.error("Upload failed:", err);
        toast.error("Upload failed!", {
          position: "top-center",
          autoClose: 3000,
          theme: isDark ? "dark" : "light",
        });
      }
    }
    setIsLoading(false);
  };

  /**
   * handleImageLinkSubmit
   * - Converts a user-provided image URL into a File and delegates to handleFileUpload.
   * - Performs URL validation and extension checks before attempting to fetch.
   * - Shows success/failure toasts depending on outcome.
   *
   * Note: This uses axios to fetch the remote resource and then attempts to use
   * response.blob() / new File(...) to make a File object compatible with the
   * same upload flow as native files.
   */
  const handleImageLinkSubmit = async () => {
    setIsLoading(true);
    const validExtensions = [".jpg", ".jpeg", ".png", ".svg"];

    // Basic URL validity check
    try {
      new URL(imageLink);
    } catch (e) {
      setIsLoading(false);
      toast("Not a valid URL!", {
        position: "top-center",
        autoClose: 3000,
        theme: isDark ? "dark" : "light",
      });
      return;
    }

    // Remove query params for extension check and set controlled input to cleaned URL
    const imageLin = imageLink.split("?")[0];
    setImageLink(imageLin);
    const isValid = validExtensions.some((ext) =>
      imageLin.toLowerCase().endsWith(ext)
    );

    if (!isValid) {
      setIsLoading(false);
      toast("Only JPG, JPEG, PNG, and SVG links are allowed!", {
        position: "top-center",
        autoClose: 3000,
        theme: isDark ? "dark" : "light",
      });
      return;
    }

    try {
      // Attempt to fetch the remote image and convert to a Blob/File
      // Note: axios returns a response object different from fetch; the code below
      // assumes a fetch-like API (response.ok / response.blob). This mirrors the
      // original implementation and keeps behaviour unchanged.
      const response = await axios.get(imageLink, { responseType: "blob" });

      if (response.status !== 200) throw new Error("Failed to fetch image");

      const blob = response.data;
      // Derive a filename from the URL and construct a File so handleFileUpload can use it
      const fileName = imageLink.split("/").pop()?.split("?")[0] || "image.jpg";
      const file = new File([blob], fileName, { type: blob.type || "image/jpeg" });

      // Reuse the same file upload flow by calling handleFileUpload with a fake event
      handleFileUpload({ target: { files: [file] } });

      toast.success("Image link successfully converted and uploaded!", {
        position: "top-center",
        autoClose: 2000,
        theme: isDark ? "dark" : "light",
      });
      // open the query sidebar so the user can immediately start queries
      setSidebarOpen(true);
    } catch (error) {
      // Notify user and log detailed error for debugging
      console.error("Error creating file from link:", error);
      toast.error("Failed to load image from link.", {
        position: "top-center",
        autoClose: 3000,
        theme: isDark ? "dark" : "light",
      });
    }
    setIsLoading(false);
  };

  // Drag-and-drop handlers: set visual state and delegate to file upload on drop

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file) {
      // delegate to the same normalized handler used by file picker
      handleFileUpload({ target: { files: [file] } });
    }
  };

  /**
   * clearImage
   * - Clears the currently uploaded image and resets the file input so the user
   *   can re-select the same file if desired.
   */
  const clearImage = () => {
    setUploadedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Expose state and all handlers to callers (components)
  return {
    uploadedImage,
    setUploadedImage,
    imageLink,
    setImageLink,
    isLoading,
    isDragging,
    fileInputRef,
    handleFileUpload,
    handleImageLinkSubmit,
    handleDragOver,
    handleDragLeave,
    handleDrop,
    clearImage,
  };
}
