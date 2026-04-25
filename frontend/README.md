# Frontend — Image Analysis Interface

This frontend provides an elegant, fast, and modern interface for image-based analysis workflows. Built with React, Tailwind CSS, GSAP, and Motion, the UI focuses on smooth animations, intuitive interactions, and a clean dark/light visual identity.

Users can upload images, run AI-powered queries, manage multiple analysis sessions, and browse past activity through an intelligent session-based sidebar.

## ✨ Key Features

### 🎨 Modern, Adaptive UI (Dark/Light Mode)
The interface adapts to the user’s theme preference, providing:
- Soft gradients and glassmorphism in light mode
- Neon accents and subtle glows in dark mode
- Smooth transitions and motion-driven UI responses

### 🖼️ Image Upload & Preview
- Beautifully framed uploaded images with:
- Auto-fallback image handling
- Adaptive cards based on theme
- Animations for mount/unmount

### 🧠 Analysis Workflow
Users can:
- Upload an image
- Enter multiple queries
- Receive structured AI results
- View, delete, or revisit past interactions

Each analysis session stores:
- Timestamp
- Uploaded image preview
- Query result count

### 📚 Chat/Analysis History
A dedicated sliding sidebar provides:
- Grouped sessions (Today, Yesterday, Last 7 Days, Older)
- Animated session items
- Quick access to previously uploaded images and their analysis
- Ability to delete or create new sessions

### 🌌 Background Effects
The interface includes:
- Animated grid backgrounds
- Floating particles
- Parallax-driven space imagery
- Mode-aware glow blobs

All effects are GPU-optimized and unobtrusive.

## 📁 Component Structure
<pre>
src/
│
├── components/
│   ├── BackgroundEffects.jsx
│   ├── ChatHistoryPanel.jsx
│   ├── Header.jsx
│   ├── ImageUploadArea.jsx
│   ├── ImageWithFallback.jsx
│   ├── QueryResultList.jsx
│   ├── QuerySidebar.jsx
│   ├── UploadedImageDisplay.jsx
│   └── ui/
│ 
├── pages/
│   ├── AuthPage.jsx
│   ├── LandingPage.jsx
│   └── MainInterface.jsx
│
├── hooks/
│   ├── useAuth.js
│   ├── useChatManagement.js
│   ├── useImageUpload.js
│   └── useQueryManagement.js
│
├── App.jsx 
└── main.jsx
</pre>

## 🚀 User Experience Goals
The UI is designed to feel:
- Fast — all transitions are motion-optimized
- Minimal yet rich — clean layout with meaningful use of color
- Intuitive — actions visible, predictable, and reversible
- Non-intrusive — background effects never interfere with the workflow

Each component contributes to a cohesive visual language that guides the user naturally through the image-analysis pipeline.

## 📦 Tech Stack
- React 18
- Tailwind CSS
- Motion (Framer Motion v3 API)
- GSAP
- Lucide Icons