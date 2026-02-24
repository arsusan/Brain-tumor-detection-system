---
title: NeuroScan AI
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

---

# NeuroScan AI: Advanced Brain Tumor Diagnostic System

NeuroScan AI is a full-stack, medical-imaging diagnostic platform that leverages Deep Learning to classify brain MRI scans. The system provides real-time analysis across four distinct categories with integrated Explainable AI (XAI) using Grad-CAM heatmaps to assist clinical decision-making.

## System Overview

The project is architected as a **Three-Tier Clinical System**:

- **Presentation Tier:** Next.js (React) frontend deployed on Vercel.
- **Logic Tier:** FastAPI (Python) backend with a Convolutional Neural Network (CNN) hosted on Hugging Face.
- **Data Tier:** PostgreSQL (Supabase) for clinical records and Cloudinary for medical image storage.

---

## Diagnostic Capabilities

The AI model is trained to identify and classify the following:

1. **Glioma:** Detection of primary brain tumors.
2. **Meningioma:** Identification of tumors in the meninges.
3. **Pituitary:** Classification of pituitary gland abnormalities.
4. **No Tumor:** Confirmation of clear scans.

---

## Technical Stack

### **Frontend (The Interface)**

- **Framework:** Next.js 14+ (App Router)
- **Styling:** Tailwind CSS (Clinical-grade dark/light theme)
- **Icons:** Lucide-React
- **PDF Engine:** jsPDF (Automated medical report generation)
- **API Client:** Axios (Customized for Multipart/Form-data)

### **Backend (The Intelligence)**

- **Framework:** FastAPI
- **AI Framework:** TensorFlow / Keras
- **Logic:** Custom CNN Architecture with **Grad-CAM** integration for activation mapping.
- **Database ORM:** SQLAlchemy
- **Environment:** Containerized deployment on Hugging Face Spaces.

### **Infrastructure**

- **Database:** Supabase (PostgreSQL)
- **Image Hosting:** Cloudinary (Secure medical imagery storage)
- **CI/CD:** GitHub Actions / Vercel Integration

---

## Key Technical Challenges & Solutions

### **Architecture Reconstruction**

To solve versioning conflicts between training and production environments, the backend implements **Weight Injection**. Instead of loading the entire model file, the system builds the model "skeleton" from code and injects the learned weights, bypassing Keras metadata version errors.

### **Explainable AI (Grad-CAM)**

The system uses **Gradient-weighted Class Activation Mapping** to highlight the morphological features the AI uses for its decision. This transforms a "Black Box" model into a transparent tool for doctors.

### **The Hybrid Image Pipeline**

To maintain high performance and low latency:

1. Frontend uploads MRI to the FastAPI backend.
2. Backend stores original image in Cloudinary.
3. CNN processes the scan and generates a diagnostic heatmap.
4. Heatmap is stored in Cloudinary; metadata is saved to PostgreSQL.
5. Frontend displays results and allows for immediate PDF download.

---

## Project Structure

```text
├── backend/
│   ├── main.py            # FastAPI entry point & API logic
│   ├── database.py        # SQLAlchemy models & Supabase connection
│   ├── explainability.py  # Grad-CAM implementation
│   └── research/          # Model architecture & Preprocessing logic
├── frontend/
│   ├── app/               # Next.js Pages & UI Components
│   ├── lib/api.ts         # Axios configuration
│   └── public/            # Static assets
└── models/                # Trained .keras weight files

```

---

## ⚙️ Detailed Deployment Instructions

### 1. Backend Deployment (Hugging Face Spaces)

Hugging Face Spaces provides the infrastructure to run your FastAPI application and TensorFlow model in a containerized environment.

#### **Step 1: Space Creation**

- Create a new **Space** on Hugging Face.
- Select **Docker** as the SDK (or **Python** if using their native template).
- Set the visibility to **Public** (required for the free tier to remain active).

#### **Step 2: Environment Secrets**

Navigate to **Settings > Variables and Secrets** in your Space and add the following:

- **`CLOUDINARY_CLOUD_NAME`**: Found in your Cloudinary Dashboard.
- **`CLOUDINARY_API_KEY`**: Your 15-digit API Key.
- **`CLOUDINARY_API_SECRET`**: Your private API Secret.
- **`DATABASE_URL`**: Your Supabase connection string (use the **Pooler** connection string for serverless environments).
- **`TESTING`**: Set to `False` to ensure the real model is loaded.
- **`PORT`**: Set to `7860` (the default port Hugging Face expects).

#### **Step 3: Dependency Management**

Ensure your `requirements.txt` includes these specific versions to avoid the Keras deserialization issues we solved earlier:

```text
fastapi
uvicorn
python-multipart
tensorflow==2.15.0
keras==2.15.0
cloudinary
sqlalchemy
psycopg2-binary
python-dotenv

```

---

### 2. Frontend Deployment (Vercel)

Vercel handles the build process for your Next.js application and serves it globally via their Edge Network.

#### **Step 1: GitHub Integration**

- Push your local code to a GitHub repository.
- In the Vercel Dashboard, click **New Project** and select your repository.

#### **Step 2: Root Directory & Build Settings**

- **Root Directory**: If your frontend code is in a subfolder (e.g., `/frontend`), click "Edit" and select that folder.
- **Framework Preset**: Ensure **Next.js** is selected.

#### **Step 3: Environment Variables**

This is the most critical step for connecting your tiers. Add the following under **Project Settings > Environment Variables**:

- **Key**: `NEXT_PUBLIC_API_URL`
- **Value**: `https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space`
- _Note: Do not include a trailing slash._

---

### 3. Database & Storage Configuration

Before the first scan, ensure your external services are ready:

- **Supabase**: Run the `init_db()` function or manually execute your SQL schema to create the `users` and `scan_results` tables.
- **Cloudinary**: Create two folders in your Media Library named `neuroscan/uploads` and `neuroscan/results` to keep your storage organized.

---

### 4. Verification Workflow

After both deployments are complete:

1. **Backend Check**: Visit `https://your-space.hf.space/`. You should see the JSON response: `{"status": "success", "message": "NeuroScan AI Backend is Running"}`.
2. **Frontend Check**: Visit your Vercel URL and attempt a "Test Scan."
3. **Log Monitoring**: If an error occurs, check the **Space Logs** on Hugging Face for model errors and the **Vercel Logs** for connection errors.

## Clinical Disclaimer

_This software is a proof-of-concept for research purposes. All AI-generated reports should be verified by a board-certified radiologist._

---
