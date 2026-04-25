import express from "express";
import multer from "multer";
import path from "path";
import fs from "fs";
import { backendLink } from "../../config.js";

const router = express.Router();

// Ensure uploads directory exists on disk before multer writes files.
// Creates the directory recursively if it does not exist (safe on startup).
const uploadDir = path.resolve("uploads");
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

// Configure multer disk storage:
// - destination: where to store uploaded files
// - filename: generate a simple unique name using timestamp + original extension
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}${path.extname(file.originalname)}`;
    cb(null, uniqueName);
  },
});

// Multer instance used to parse multipart/form-data requests with single "image" field
const upload = multer({ storage });

// POST /api/upload
// - Accepts a single file under form field "image"
// - On success returns a JSON object with an accessible URL to the stored file
router.post("/", upload.single("image"), (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });

  const fileName = req.file.filename;
  // Build the public URL that the frontend can use to access the uploaded file
  const imageUrl = `${backendLink}/api/uploads/${fileName}`;

  res.json({ imageUrl });
});

export default router;
