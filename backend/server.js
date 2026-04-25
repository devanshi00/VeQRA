import express from "express";
import cors from "cors";
import authRoutes from "./routes/auth.route.js";
import chatRoutes from "./routes/chat.route.js";
import queryRoutes from "./routes/query.route.js";
import uploadRoutes from "./routes/upload.route.js";
import { frontendLink } from "../config.js";

/**
 * server.js
 * - Entry point for the backend API server.
 * - Sets up express, enables CORS for the frontend origin, mounts route modules
 *   and serves static folders for uploaded assets and generated results.
 */

const app = express();

// Parse incoming JSON payloads
app.use(express.json());

// Allow requests only from configured frontend origin (simple CORS policy)
app.use(cors({ origin: `${frontendLink}` }));

// Mount modular route handlers under /api/*
app.use("/api/auth", authRoutes);       // auth: login/signup endpoints
app.use("/api/chat", chatRoutes);       // chat: create/list/delete chat sessions
app.use("/api/query", queryRoutes);     // query: VQA / grounding endpoints
app.use("/api/upload", uploadRoutes);   // upload: image upload endpoints

// Expose filesystem directories for serving uploaded files and result artifacts
// - /api/uploads -> serves files from backend/uploads
// - /api/results -> serves generated result files (e.g., image outputs)
app.use("/api/uploads", express.static("uploads"));
app.use("/api/results", express.static("results"));

// Start listening on port 5000 and log readiness
app.listen(5000, () => console.log("✅ Server running on port 5000"));
