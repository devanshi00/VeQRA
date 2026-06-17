# Generative AI for Natural Language Understanding of Satellite Imagery
## VeQRA — Visual Earth Query and Retrieval Assistant

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Stack](https://img.shields.io/badge/Node.js-339933?logo=node.js&logoColor=white)
![Stack](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Stack](https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black)
![Stack](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)

**VeQRA** is an advanced AI-powered platform designed for multimodal inference on earth observation and general imagery. It seamlessly integrates image captioning, visual grounding, translation, and query-based analysis into a high-performance web interface.

## Table of Contents

- [Overview](#overview)
- [Core Features](#core-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Research & Reproducibility](#research--reproducibility)
- [Installation & Setup](#installation--setup)
- [API Documentation](#api-documentation)
- [Unified Evaluation Mode](#unified-evaluation-mode)

## Overview

VeQRA bridges the gap between complex computer vision tasks and user interaction. It utilizes a **Node.js** and **Python** backend for heavy inference orchestration and a **React-based** frontend for the user experience.

All interactions, including chat history and image analysis results, are persisted via **PostgreSQL**, ensuring a continuous and stateful workflow.

## Solution Architecture
<img width="633" height="362" alt="Screenshot 2026-04-21 at 12 59 20 PM" src="https://github.com/user-attachments/assets/8c1f02fe-0a95-4ce0-8d6b-7c6381e5f414" />

## Core Features

### Backend (Intelligence & Logic)
- **Multimodal Inference**: Supports Captioning, Visual Question Answering (VQA), and YOLO-based Visual Grounding.
- **Data Persistence**: Robust PostgreSQL storage for chat history and query logs.
- **Microservices Architecture**: Python-based inference engines decoupled from the Node.js core.
- **Static File Management**: Efficient serving of uploaded images and processed grounding results.

### Frontend (User Experience)
- **Responsive UI**: Features modern transitions and layout.
- **Theme-Aware**: Supports Dark and Light modes.
- **Visual Query Interface**: Drag-and-drop image uploads with immediate preview and result visualization.
- **Session Management**: Organized history grouping by date.

<img width="831" height="231" alt="Screenshot 2026-04-25 at 10 15 44 PM" src="https://github.com/user-attachments/assets/79df4780-e20f-4a00-9818-383b5ebb8c11" />

## Tech Stack

| Component | Technologies |
|-----------|--------------|
| **Frontend** | React 18, Tailwind CSS, GSAP, Framer Motion |
| **Backend** | Node.js, Express, Python (Inference), PostgreSQL |
| **Storage** | Local filesystem (Uploads/Results), PostgreSQL (Metadata) |

## Project Structure

```bash
VeQRA/
│
├── backend/                                # API & Inference Logic
│   ├── routes/                             # Endpoint definitions
│   ├── uploads/                            # Raw image storage
│   ├── results/                            # Processed image storage (Grounding)
│   └── server.js                           # Entry point
│
├── frontend/                               # Client Application
│   ├── public/                             # Static assets
│   ├── src/
│   │   ├── components/                     # Reusable UI elements
│   │   └── pages/                          # Route views
│   └── index.html
│
└── scripts/                                # Research & Reproducibility
    ├── benchmarking/                       # Baseline model evaluation
    │   ├── evaluate.py                     # Metric calculation (BERT-BLEU)
    │   ├── llavanext.py                    # LLaVA-NeXT inference
    │   ├── qwenvl.py                       # Qwen-VL inference
    │   ├── rsllava.py                      # RS-LLaVA inference
    │   └── run.sh                          # Execution entry point
    ├── dataset/                            # Data ingestion & formatting
    │   ├── convert_detection.py            # Format MMRS-1M detection
    │   ├── convert_grounding.py            # Format MMRS-1M grounding
    │   ├── download_mmrs1m.sh              # Download MMRS-1M parts
    │   ├── download_vrsbench.sh            # Download VRSBench
    │   └── mmrs1m_structure.py             # Structure mapping utility
    └── finetuning/                         # Training pipelines (LoRA)
        ├── evaluate.py                     # Evaluation logic
        ├── evaluate.sh                     # Evaluation execution
        ├── finetune_caption.sh             # Captioning pipeline entry
        ├── finetune_caption.py             # Captioning training script
        ├── inference_caption.py            # Captioning validation script
        ├── finetune_vqa.sh                 # VQA pipeline entry
        ├── finetune_vqa.py                 # VQA training script
        ├── inference_vqa.py                # VQA validation script
        ├── finetune_hybrid.sh              # Hybrid pipeline entry
        ├── finetune_hybrid.py              # Hybrid training script
        ├── inference_hybrid.py             # Hybrid validation script
        ├── multimodality_pipeline.py       # SAR/IR adapter training
        ├── skysense_test.sh                # SkySense benchmark entry
        └── skysense_test.py                # SkySense inference
```

## Research & Reproducibility

The `scripts/` directory contains the complete pipeline used to train, validate, and benchmark the models underlying VeQRA. Follow the steps below to reproduce the results.

### 1\. Dataset Preparation

Before training, datasets must be downloaded and formatted into the expected JSON structure.

```bash
cd scripts/dataset

# Download VRSBench and MMRS-1M datasets
bash download_vrsbench.sh
bash download_mmrs1m.sh

# Convert raw annotations to training format
python3 convert_detection.py
python3 convert_grounding.py
```

### 2\. Fine-Tuning Pipelines

We provide specific shell scripts that handle environment setup, dependency installation, and the execution of training scripts. These scripts use QLoRA for memory-efficient fine-tuning.

Navigate to the fine-tuning directory:

```bash
cd scripts/finetuning
```

**Task-Specific Pipelines:**

  * **Captioning Task:**
    Trains Qwen2.5-VL on VRSBench captioning data.

    ```bash
    bash finetune_caption.sh
    ```

  * **Visual Question Answering (VQA):**
    Trains on VRSBench VQA pairs.

    ```bash
    bash finetune_vqa.sh
    ```

  * **Hybrid Training:**
    Trains on a combined dataset of VRSBench and SkySense to improve generalization.

    ```bash
    bash finetune_hybrid.sh
    ```

  * **Modality-Specific Adapters:**
    Trains adapters specifically for SAR or IR modalities using the MMRS-1M dataset.

    ```bash
    # For SAR Modality
    python3 multimodality_pipeline.py --modality sar

    # For IR Modality
    python3 multimodality_pipeline.py --modality ir
    ```

**Validation:**
To evaluate the fine-tuned models and generate metrics:

```bash
bash evaluate.sh
```

**External Benchmarking:**
To run the SkySense benchmark test:

```bash
bash skysense_test.sh
```

### 3\. Benchmarking & Evaluation

To compare the fine-tuned models against baselines (e.g., LLaVA-NeXT, Qwen-VL, RS-LLaVA), utilize the benchmarking suite.

```bash
cd scripts/benchmarking

# Run inference on all baseline models and generate metrics
bash run.sh
```

## Installation & Setup (Web Application)

### Prerequisites

  - Node.js (v18+)
  - Python (v3.9+)
  - PostgreSQL Database

### Backend Setup

1.  **Navigate to the backend directory:**

    ```bash
    cd backend
    ```

2.  **Install dependencies:**

    ```bash
    npm install
    ```

3.  **Initialize storage directories:**

    ```bash
    mkdir uploads results
    ```

4.  **Configuration:**
    Update `config.js` with your PostgreSQL credentials and Python environment paths.

5.  **Start the server:**

    ```bash
    node server.js
    ```

## API Documentation

### Authentication

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/auth/signup` | Register a new user account. |
| `POST` | `/api/auth/login` | Authenticate and retrieve session token. |

### Media Management

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/upload` | Upload a single image via `multipart/form-data`. |
| `GET` | `/api/uploads/:filename` | Retrieve raw uploaded image. |
| `GET` | `/api/results/:filename` | Retrieve processed grounding output. |

### Inference & Query

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/query/captioning` | Generates a textual description of the image. |
| `POST` | `/api/query/grounding` | Performs YOLO object detection and returns bounding boxes. |
| `POST` | `/api/query/vqa` | Processes a natural language query regarding the image. |
| `POST` | `/api/translate` | Translates text between supported languages. |

## Unified Evaluation Mode

For automated benchmarking and structured reasoning, VeQRA offers a **Unified Multimodal Evaluation API**. This endpoint acts as an orchestrator, chaining multiple inference tasks (Captioning, Grounding, VQA) into a single request-response cycle.

### Endpoint

`POST /api/evaluate`

### Workflow

1.  **Captioning**: Generates a scene description.
2.  **Grounding**: Identifies objects with Oriented Bounding Boxes (OBB).
3.  **Attribute Reasoning**: Executes specific VQA sub-queries (Binary, Numeric, Semantic).
4.  **Aggregation**: Returns a monolithic JSON response containing all insights.

### Request Format

The endpoint accepts a structured JSON payload defining the image metadata and the specific queries to run.

```json
{
  "input_image": {
    "image_id": "img_001",
    "image_url": "http://source/path/to/image.jpg",
    "metadata": {
      "width": 1024,
      "height": 1024,
      "spatial_resolution_m": 0.5
    }
  },
  "queries": {
    "caption_query": {
      "instruction": "Detailed description of visible elements..."
    },
    "grounding_query": {
      "instruction": "Locate and return oriented bounding boxes..."
    },
    "attribute_query": {
      "binary": {
        "instruction": "Is there a vehicle present? (Answer: Yes/No)"
      },
      "numeric": {
        "instruction": "Count the number of buildings. (Answer: Float)"
      },
      "semantic": {
        "instruction": "What is the terrain type? (Answer: Text)"
      }
    }
  }
}
```

### Purpose

  - **Standardization**: Ensures consistent output formats for benchmarking.
  - **Efficiency**: Reduces network overhead by consolidating multiple API calls.
  - **Automation**: Designed for batch processing large datasets.

