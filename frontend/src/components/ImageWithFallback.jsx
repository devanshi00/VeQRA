import { useState, useEffect } from "react";
import { createPortal } from "react-dom";

/**
 * ImageWithFallback
 * - Renders an image with:
 *   - automatic fallback SVG if the image fails to load
 *   - clickable fullscreen preview (esc or click outside to close)
 * - Keeps body scroll locked while preview is open.
 */

/* Inline SVG fallback encoded as a data URI used when the image errors */
const ERROR_IMG_SRC =
  "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODgiIGhlaWdodD0iODgiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgc3Ryb2tlPSIjMDAwIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBvcGFjaXR5PSIuMyIgZmlsbD0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIzLjciPjxyZWN0IHg9IjE2IiB5PSIxNiIgd2lkdGg9IjU2IiBoZWlnaHQ9IjU2IiByeD0iNiIvPjxwYXRoIGQ9Im0xNiA1OCAxNi0xOCAzMiAzMiIvPjxjaXJjbGUgY3g9IjUzIiBjeT0iMzUiIHI9IjciLz48L3N2Zz4KCg==";

export function ImageWithFallback(props) {
  // local UI state
  const [didError, setDidError] = useState(false); // whether original src failed to load
  const [isPreviewOpen, setIsPreviewOpen] = useState(false); // fullscreen preview open state

  // accept typical <img> props; extract common ones and keep the rest
  const { src, alt, style, className, ...rest } = props;

  // handlers
  const handleError = () => setDidError(true); // swap to fallback on error
  const handleClick = () => {
    // open preview only if original image loaded (or we still want to preview src)
    if (!didError) setIsPreviewOpen(true);
  };
  const closePreview = () => setIsPreviewOpen(false);

  // close preview on Escape key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") closePreview();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // lock body scroll when preview is open (prevents background scroll)
  useEffect(() => {
    if (isPreviewOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      // cleanup in case component unmounts while locked
      document.body.style.overflow = "";
    };
  }, [isPreviewOpen]);

  return (
    <>
      {/* Primary image element:
          - shows fallback data URI when an error occurs
          - clickable to open fullscreen preview
      */}
      <img
        src={didError ? ERROR_IMG_SRC : src}
        alt={didError ? "Error loading image" : alt}
        className={`${className ?? ""} cursor-pointer`}
        style={style}
        onError={handleError}
        onClick={handleClick}
        {...rest}
      />

      {/* Fullscreen preview rendered into document.body via portal.
          Clicking outside the image or pressing Escape closes it.
          The close button is accessible (aria-label) and stops propagation.
      */}
      {isPreviewOpen &&
        createPortal(
          <div
            className="fixed inset-0 z-9999 flex items-center justify-center bg-black/10 backdrop-blur-sm"
            onClick={closePreview}
            style={{ margin: 0, padding: 0 }}
          >
            {/* Prevent clicks on the image from closing the preview */}
            <img
              src={src}
              alt={alt}
              className="max-w-[90vw] max-h-[85vh] w-auto h-auto object-contain"
              onClick={(e) => e.stopPropagation()}
              style={{ maxWidth: "100vw", maxHeight: "100vh" }}
            />

            {/* Close button (visible in preview) */}
            <button
              className="top-1 right-1 text-white text-4xl font-bold bg-black/50 rounded-full w-12 h-12 flex items-center justify-center hover:bg-black/70 transition"
              onClick={(e) => {
                e.stopPropagation();
                closePreview();
              }}
              aria-label="Close preview"
            >
              ×
            </button>
          </div>,
          document.body
        )}
    </>
  );
}
