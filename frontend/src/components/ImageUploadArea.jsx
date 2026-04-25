import { motion } from "motion/react";
import { Card } from "./ui/card";
import { Button } from "./ui/button";
import { Upload, ImageIcon, Link } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";

/**
 * ImageUploadArea
 * - Tabbed UI that lets the user provide a satellite image via a direct link or file upload.
 * - Props:
 *   - isDark: boolean for theme styling
 *   - isDragging: boolean whether a drag-over is active (changes copy/styles)
 *   - imageLink: controlled input value for the link tab
 *   - isLoading: disables actions while processing
 *   - fileInputRef: ref to hidden file input (used to trigger file picker)
 *   - onFileUpload: handler for file input change
 *   - onDragOver / onDragLeave / onDrop: drag-and-drop handlers for file area
 *   - onImageLinkChange: controlled input updater for link field
 *   - onImageLinkSubmit: handler to submit the provided image URL
 */
export function ImageUploadArea({
  isDark,
  isDragging,
  imageLink,
  isLoading,
  fileInputRef,
  onFileUpload,
  onDragOver,
  onDragLeave,
  onDrop,
  onImageLinkChange,
  onImageLinkSubmit,
}) {
  return (
    <Tabs defaultValue="upload" className="w-full">
      {/* Tab switcher: Link vs Upload */}
      <TabsList className="grid w-full grid-cols-2 bg-white/5 rounded-t-2xl">
        <TabsTrigger
          value="upload"
          className={`rounded-xl data-[state=active]:bg-linear-to-r data-[state=active]:from-[#0077b6] data-[state=active]:to-[#48cae4] ${
            isDark ? "text-white" : "data-[state=active]:text-white"
          } data-[state=active]:shadow-lg transition-all`}
        >
          Upload
        </TabsTrigger>
        <TabsTrigger
          value="link"
          className={`rounded-xl data-[state=active]:bg-linear-to-r data-[state=active]:from-[#0077b6] data-[state=active]:to-[#48cae4] ${
            isDark ? "text-white" : "data-[state=active]:text-white"
          } data-[state=active]:shadow-lg transition-all`}
        >
          Link
        </TabsTrigger>
      </TabsList>

      {/* Link tab: user pastes a direct image URL */}
      <TabsContent value="link" className="sm:p-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Card
            className={`p-12 rounded-2xl border transition-all duration-300 ${
              isDark
                ? "bg-white/5 border-white/10 backdrop-blur-xl hover:bg-white/10"
                : "bg-white border-[#e3edf3] hover:shadow-xl"
            }`}
          >
            <div className="text-center space-y-6">
              {/* Decorative icon block */}
              <motion.div
                animate={{ y: [-5, 5, -5] }}
                transition={{
                  duration: 3,
                  repeat: Infinity,
                  ease: "easeInOut",
                }}
                className={`w-24 h-24 mx-auto rounded-3xl flex items-center justify-center relative ${
                  isDark
                    ? "bg-linear-to-br from-[#48cae4]/20 to-[#0077b6]/20"
                    : "bg-linear-to-br from-[#e3edf3] to-[#c7d4de]"
                }`}
              >
                <div
                  className={`absolute inset-0 rounded-3xl blur-xl ${
                    isDark ? "bg-[#48cae4]/30" : "bg-[#0077b6]/20"
                  }`}
                />
                <Link
                  className={`w-12 h-12 relative z-10 ${
                    isDark ? "text-[#48cae4]" : "text-[#0077b6]"
                  }`}
                />
              </motion.div>

              {/* Title + help text */}
              <div>
                <h2
                  className={`mb-2 ${isDark ? "text-white" : "text-[#1b263b]"}`}
                >
                  Enter Satellite Image Link
                </h2>
                <p className={isDark ? "text-[#8d99ae]" : "text-[#43515f]"}>
                  Paste a direct image URL (jpg, jpeg, png, svg)
                </p>
              </div>

              {/* Input + submit button */}
              <div className="flex flex-col items-center justify-center gap-4 w-full">
                <input
                  type="text"
                  placeholder="https://example.com/image.jpg"
                  value={imageLink}
                  onChange={(e) => onImageLinkChange(e.target.value)}
                  className={`w-full sm:w-2/3 px-4 py-4 rounded-xl outline-none border transition-all duration-300 ${
                    isDark
                      ? "bg-white/5 border-white/10 text-white placeholder:text-[#8d99ae] focus:border-[#48cae4]"
                      : "bg-[#f9fbfd] border-[#cbd5e1] text-[#1b263b] placeholder:text-[#94a3b8] focus:border-[#0077b6]"
                  }`}
                />
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Button
                    disabled={isLoading}
                    onClick={onImageLinkSubmit}
                    className={`px-8 py-6 rounded-xl shadow-xl hover:shadow-2xl transition-all duration-300 relative overflow-hidden group ${
                      isDark
                        ? "bg-linear-to-r from-[#48cae4] to-[#00b4d8] hover:from-[#48cae4] hover:to-[#0096c7] text-[#0d1b2a]"
                        : "bg-linear-to-r from-[#0077b6] to-[#005f8f] hover:from-[#0077b6] hover:to-[#0099cc] text-white"
                    }`}
                  >
                    {/* Animated sheen overlay */}
                    <div className="absolute inset-0 bg-linear-to-r from-white/0 via-white/20 to-white/0 translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-1000" />
                    <span className="relative flex items-center gap-2">
                      <ImageIcon className="w-5 h-5" />
                      {isLoading ? "Processing..." : "Use Image Link"}
                    </span>
                  </Button>
                </motion.div>
              </div>
            </div>
          </Card>
        </motion.div>
      </TabsContent>

      {/* Upload tab: drag-and-drop or file picker */}
      <TabsContent value="upload" className="p-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Card
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            className={`p-12 rounded-2xl border transition-all duration-300 cursor-pointer ${
              isDark
                ? `bg-white/5 border-white/10 backdrop-blur-xl hover:bg-white/10 ${
                    isDragging ? "border-[#48cae4]" : ""
                  }`
                : `bg-white border-[#e3edf3] hover:shadow-xl ${
                    isDragging ? "border-[#0077b6]" : ""
                  }`
            }`}
          >
            <div className="text-center space-y-6">
              {/* Decorative upload icon block (animates subtly) */}
              <motion.div
                animate={{ y: [-5, 5, -5] }}
                transition={{
                  duration: 3,
                  repeat: Infinity,
                  ease: "easeInOut",
                }}
                className={`w-24 h-24 mx-auto rounded-3xl flex items-center justify-center relative ${
                  isDark
                    ? "bg-linear-to-br from-[#48cae4]/20 to-[#0077b6]/20"
                    : "bg-linear-to-br from-[#e3edf3] to-[#c7d4de]"
                }`}
              >
                <div
                  className={`absolute inset-0 rounded-3xl blur-xl ${
                    isDark ? "bg-[#48cae4]/30" : "bg-[#0077b6]/20"
                  }`}
                />
                <Upload
                  className={`w-12 h-12 relative z-10 ${
                    isDark ? "text-[#48cae4]" : "text-[#0077b6]"
                  }`}
                />
              </motion.div>

              {/* Title and drag instructions (changes when dragging) */}
              <div>
                <h2
                  className={`mb-2 ${isDark ? "text-white" : "text-[#1b263b]"}`}
                >
                  {isDragging ? "Drop image here" : "Upload Satellite Image"}
                </h2>
                <p className={isDark ? "text-[#8d99ae]" : "text-[#43515f]"}>
                  {isDragging
                    ? "Release to upload"
                    : "Drag & drop or select an image for analysis"}
                </p>
              </div>

              {/* Hidden file input triggered by the Select Image button */}
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.svg"
                onChange={onFileUpload}
                className="hidden"
              />

              {/* Button to open native file picker */}
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  className={`px-8 py-6 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-300 relative overflow-hidden group ${
                    isDark
                      ? "bg-linear-to-r from-[#48cae4] to-[#00b4d8] hover:from-[#48cae4] hover:to-[#0096c7] text-[#0d1b2a]"
                      : "bg-linear-to-r from-[#0077b6] to-[#005f8f] hover:from-[#0077b6] hover:to-[#0099cc] text-white"
                  }`}
                >
                  <div className="absolute inset-0 bg-linear-to-r from-white/0 via-white/20 to-white/0 translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-1000" />
                  <span className="relative flex items-center gap-2">
                    <ImageIcon className="w-5 h-5" />
                    Select Image
                  </span>
                </Button>
              </motion.div>
            </div>
          </Card>
        </motion.div>
      </TabsContent>
    </Tabs>
  );
}
