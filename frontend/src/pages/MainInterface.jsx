import { useState } from "react";
import { ToastContainer } from "react-toastify";
import { useAuth } from "../hooks/useAuth";
import { useImageUpload } from "../hooks/useImageUpload";
import { useQueryManagement } from "../hooks/useQueryManagement";
import { useChatManagement } from "../hooks/useChatManagement";
import { BackgroundEffects } from "../components/BackgroundEffects";
import { ChatHistoryPanel } from "../components/ChatHistoryPanel";
import { Header } from "../components/Header";
import { ImageUploadArea } from "../components/ImageUploadArea";
import { UploadedImageDisplay } from "../components/UploadedImageDisplay";
import { QueryResultsList } from "../components/QueryResultList";
import { QuerySidebar } from "../components/QuerySidebar";
import { backendLink } from "../../../config.js";

/**
 * MainInterface
 * - Primary logged-in UI that composes uploads, chat history, query sidebar,
 *   image preview and query results together.
 *
 * Props:
 *  - onLogout: callback to sign the user out
 *  - theme: "dark" | "light" used to style children
 *  - onToggleTheme: toggles the app theme
 *  - username: current user's display name
 */
export function MainInterface({ onLogout, theme, onToggleTheme, username }) {
  // Sidebar / history panel visibility
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [historyPanelOpen, setHistoryPanelOpen] = useState(false);

  // boolean convenience used widely for conditional classes
  const isDark = theme === "dark";

  // Read persisted auth; onLogout will be called by hook when no valid session
  const { userData } = useAuth(onLogout);

  // Chat session management: loads sessions, select/new/delete handlers, caption generation
  const {
    chatSessions,
    setChatSessions,
    currentChatId,
    caption,
    isGenerating,
    generateCaption,
    setCurrentChatId,
    handleSelectChat,
    handleNewChat,
    handleDeleteChat,
  } = useChatManagement(userData, isDark, setSidebarOpen);

  // Query management: input state, run handlers, results storage for the active chat
  const {
    query,
    setQuery,
    isProcessing,
    queryResults,
    setQueryResults,
    handleRunQuery,
    groundingQuery,
    setGroundingQuery,
    isGroundingProcessing,
    setIsGroundingProcessing,
    handleRunGrounding,
    queryType,
    setQueryType,
  } = useQueryManagement(currentChatId, setChatSessions, theme);

  // Image upload handling: file/link upload, drag state, and uploaded image object
  const {
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
  } = useImageUpload(
    userData,
    isDark,
    setChatSessions,
    setCurrentChatId,
    setSidebarOpen
  );

  // Download a PDF/CSV report for the currently selected chat session (opens in new tab)
  const onDownloadReport = () => {
    window.open(`${backendLink}/api/chat/${currentChatId}/report`, "_blank");
  };

  return (
    <>
      {/* Toast container for user notifications */}
      <ToastContainer />
      <div
        className={`min-h-screen transition-colors duration-500 ${
          isDark
            ? "bg-linear-to-br from-[#0a1929] via-[#1a2f42] to-[#0d1b2a]"
            : "bg-linear-to-br from-[#e3edf3] via-[#f9fafb] to-[#c7d4de]"
        }`}
      >
        {/* Decorative background effects (stars, gradients, etc.) */}
        <BackgroundEffects isDark={isDark} />

        {/* Slide-in chat history panel (left or right depending on layout) */}
        <ChatHistoryPanel
          chatSessions={chatSessions}
          currentChatId={currentChatId}
          onSelectChat={(chatId) =>
            handleSelectChat(
              chatId,
              setUploadedImage,
              setQueryResults,
            )
          }
          onNewChat={() =>
            handleNewChat(setUploadedImage, setQueryResults, fileInputRef)
          }
          onDeleteChat={(chatId) =>
            handleDeleteChat(
              chatId,
              setUploadedImage,
              setQueryResults,
              fileInputRef
            )
          }
          isOpen={historyPanelOpen}
          onToggle={() => {
            // Toggling history should close the query sidebar to avoid overlap
            setHistoryPanelOpen(!historyPanelOpen);
            setSidebarOpen(false);
          }}
          theme={theme}
        />

        {/* Top header with menu, theme toggle, query open and logout */}
        <Header
          isDark={isDark}
          username={username}
          uploadedImage={!!uploadedImage}
          historyPanelOpen={historyPanelOpen}
          onToggleHistory={() => {
            setHistoryPanelOpen(!historyPanelOpen);
            setSidebarOpen(false);
          }}
          onToggleTheme={onToggleTheme}
          onOpenQuerySidebar={() => setSidebarOpen(true)}
          onLogout={onLogout}
        />

        {/* Main container: either upload area or image + results layout */}
        <div
          className={`relative container ${
            sidebarOpen ? "left-2" : "mx-auto"
          } px-3 md:px-6 py-12`}
        >
          <div
            className={`${
              uploadedImage && queryResults.length !== 0
                ? "max-w-5xl 2xl:max-w-6xl mx-auto"
                : "max-w-4xl mx-auto"
            }`}
          >
            {/* If no image uploaded -> show the upload UI (link or file) */}
            {!uploadedImage ? (
              <ImageUploadArea
                isDark={isDark}
                isDragging={isDragging}
                imageLink={imageLink}
                isLoading={isLoading}
                fileInputRef={fileInputRef}
                onFileUpload={handleFileUpload}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onImageLinkChange={setImageLink}
                onImageLinkSubmit={handleImageLinkSubmit}
              />
            ) : (
              /* When an image is uploaded, show preview + results area */
              <div className="flex flex-col lg:flex-row gap-6 items-start">
                {/* Spacer column to reserve space for the fixed preview on large screens */}
                {queryResults.length > 0 && <div className="hidden lg:block w-116 shrink-0" />}

                {/* Uploaded image preview (fixed on large screens when results exist) */}
                {queryResults.length > 0 && (
                  <div
                    className={`block self-center lg:fixed ${
                      sidebarOpen ? "left-20" : "lg:left-[3vw] xl:left-1/12 2xl:left-1/10"
                    } top-[10vh] max-w-xl z-9`}
                    >
                    <UploadedImageDisplay
                      isDark={isDark}
                      uploadedImage={uploadedImage}
                      caption={caption}
                      isGenerating={isGenerating}
                      generateCaption={generateCaption}
                      />
                  </div>
                )}

                {/* Query results list column */}
                {queryResults.length > 0 && (
                  <div className="flex-1 self-center max-w-3xl">
                    <QueryResultsList
                      isDark={isDark}
                      queryResults={queryResults}
                    />
                  </div>
                )}

                {/* If no results yet, center the uploaded image display */}
                {queryResults.length === 0 && (
                  <div className="max-w-3xl mx-auto">
                    <UploadedImageDisplay
                      isDark={isDark}
                      uploadedImage={uploadedImage}
                      caption={caption}
                      isGenerating={isGenerating}
                      generateCaption={generateCaption}
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Query sidebar: contains query inputs, voice controls and run buttons */}
        <QuerySidebar
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          query={query}
          onQueryChange={setQuery}
          onRunQuery={handleRunQuery}
          onRunGrounding={handleRunGrounding}
          isProcessing={isProcessing}
          onDownloadReport={onDownloadReport}
          isGroundingProcessing={isGroundingProcessing}
          setIsGroundingProcessing={setIsGroundingProcessing}
          groundingQuery={groundingQuery}
          setGroundingQuery={setGroundingQuery}
          handleRunGrounding={handleRunGrounding}
          queryType={queryType}
          setQueryType={setQueryType}
          theme={theme}
        />
      </div>
    </>
  );
}
