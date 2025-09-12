# AI Image Caption Generator

An end-to-end deep learning application that automatically generates natural language captions for uploaded images using a CNN + LSTM architecture.

## 🎯 Features

- **Real-time Image Captioning**: Upload images and get AI-generated captions instantly
- **Deep Learning Pipeline**: CNN feature extraction + LSTM sequence generation
- **Full-Stack Application**: React frontend with FastAPI backend
- **Production Ready**: Trained on 40,000+ image-caption pairs from Flickr8k dataset
- **High Accuracy**: 71% training accuracy with beam search optimization

## 🏗️ Project Structure

```
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py         # FastAPI application entry point
│   │   ├── services/       # ML services (caption generation, training)
│   │   └── utils/          # Utility functions
│   ├── scripts/            # Training and testing scripts
│   ├── tests/              # Backend tests
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile         # Backend container configuration
├── frontend/               # React frontend
│   ├── src/               # React source code
│   ├── public/            # Static assets
│   ├── Dockerfile         # Frontend container configuration
│   └── nginx.conf         # Nginx configuration
├── shared/                # Shared utilities and types
│   ├── types.ts           # TypeScript type definitions
│   └── constants.ts       # Shared constants
├── docker-compose.yml     # Local development setup
├── cloudbuild.yaml        # Google Cloud Build configuration
└── README.md             # This file
```

## 🚀 Development Setup

### Backend Setup

1. Create and activate virtual environment:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the backend:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start development server:
```bash
npm start
```

### Docker Development

Run the entire application with Docker Compose:
```bash
docker-compose up --build
```

## 🤖 Model Training

The application includes comprehensive training scripts for the image captioning model:

### Quick Training (for testing)
```bash
cd backend
python scripts/train_model_simple.py
```

### Full Dataset Training (production model)
```bash
cd backend
python scripts/train_model_full_dataset_fixed.py
```

### Model Performance
- **Dataset**: Flickr8k (40,000+ image-caption pairs)
- **Architecture**: ResNet50 (CNN) + LSTM (RNN)
- **Training Accuracy**: 71.9%
- **Validation Accuracy**: 64.0%
- **Vocabulary Size**: 8,497 words

## 📡 API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/v1/generate-caption` - Generate caption for uploaded image

## 🛠️ Technology Stack

- **Backend**: FastAPI, TensorFlow/Keras, ResNet50, LSTM
- **Frontend**: React, TypeScript, Axios
- **ML Pipeline**: CNN feature extraction, LSTM sequence generation, Beam search
- **Deployment**: Google Cloud Run, Docker
- **CI/CD**: Google Cloud Build

## 🎨 Example Captions Generated

- "a man in a black wetsuit is surfing in the ocean"
- "a black and white dog is playing with a tennis ball on the grass"
- "people are standing on a beach with a large body of water and mountains in the background"
- "a basketball player in a white shirt is holding a basketball in front of a crowd"

## 🚀 Deployment

The application is configured for deployment on Google Cloud Platform using Cloud Build and Cloud Run.

### Prerequisites
- Google Cloud Project with billing enabled
- Cloud Build and Cloud Run APIs enabled
- Docker images pushed to Google Container Registry

### Deploy
```bash
gcloud builds submit --config cloudbuild.yaml
```
