# ATS Resume Parser Engine & Microservice

Production-grade, high-throughput Resume Parser with Layout-Aware Document Ingestion, Section Segmentation, Demographics/Experience/Skills Extraction, and Automated Qdrant Vector Search Sync.

---

## 📋 System Prerequisites

Before running the Python code, your machine requires system binaries for OCR and document format conversion.

### 1. Ubuntu / Debian Linux
```bash
sudo apt update
# 1. OCR Engine for scanned PDFs/images
sudo apt install -y tesseract-ocr tesseract-ocr-eng libtesseract-dev

# 2. LibreOffice for legacy binary .doc conversion
sudo apt install -y libreoffice-writer

# 3. PDF rendering utilities
sudo apt install -y poppler-utils
```

### 2. macOS (Homebrew)
```bash
brew update
brew install tesseract
brew install --cask libreoffice
brew install poppler
```

### 3. Windows
- **Tesseract OCR:** Download installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add `C:\Program Files\Tesseract-OCR` to system `PATH`.
- **LibreOffice:** Install standard LibreOffice Windows package and ensure `soffice` is in `PATH`.

---

## 🚀 Quick Setup & Installation

Follow these 4 simple steps to set up the project on a new machine:

### Step 1: Clone and Create Virtual Environment
```bash
git clone <repository_url>
cd ats_parser

# Create Python virtual environment (Python 3.10 or 3.11 recommended)
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows (cmd/powershell):
# .\venv\Scripts\activate
```

### Step 2: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Download spaCy Language Model (Mandatory)
SpaCy's English model is used for Name, Entity, and Company extraction:
```bash
python -m spacy download en_core_web_sm
```

### Step 4: Ensure Qdrant Vector Database is Running
The parser automatically embeds and upserts extracted resumes into a Qdrant collection named `resumes`.

Start Qdrant using Docker:
```bash
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    --name qdrant-engine \
    qdrant/qdrant
```
*(If Qdrant runs on a remote host, configure `QDRANT_HOST` and `QDRANT_PORT` in your environment).*

---

## 💻 Running the Server

### Standard High-Speed CPU Mode (Recommended)
Uses Tesseract OCR for scanned documents (fastest, memory-efficient, no GPU needed):
```bash
OCR_ENGINE=tesseract python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

The API will be live at:
- **API Base:** `http://localhost:8000`
- **Swagger Documentation:** `http://localhost:8000/docs`
- **Interactive ReDoc:** `http://localhost:8000/redoc`

### Optional: Advanced Vision Mode (Surya OCR)
If you require deep-learning layout OCR with Surya:
```bash
pip install torch surya-ocr==0.14.7 pypdfium2==4.30.0
OCR_ENGINE=surya python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## 🖥️ Running the Frontend (Optional)

If using the included React/Vite dropzone UI:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` to upload resumes interactively.

---

## 📡 Key API Endpoints

- `POST /api/v1/parse-resume` - Upload a single resume (PDF, DOCX, DOC) and receive structured JSON synchronously.
- `POST /api/v1/parse-resume-async` - Background queue for bulk or heavy scanned OCR files.
- `GET /api/v1/job-status/{job_id}` - Check the status of an asynchronous parsing job.
- `POST /api/v1/parse-batch` - Upload multiple resumes simultaneously with bounded concurrency.
