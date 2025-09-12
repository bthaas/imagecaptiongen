# AI Image Caption Generator

An end-to-end deep learning application that automatically generates natural language captions for uploaded images using a CNN + LSTM architecture.

## Project Structure

```
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py         # FastAPI application entry point
│   │   ├── models/         # ML models
│   │   ├── services/       # Business logic services
│   │   └── utils/          # Utility functions
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

## Development Setup

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

## Deployment

The application is configured for deployment on Google Cloud Platform using Cloud Build and Cloud Run.

### Prerequisites
- Google Cloud Project with billing enabled
- Cloud Build and Cloud Run APIs enabled
- Docker images pushed to Google Container Registry

### Deploy
```bash
gcloud builds submit --config cloudbuild.yaml
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/v1/generate-caption` - Generate caption for uploaded image

## Technology Stack

- **Backend**: FastAPI, TensorFlow, Python 3.9
- **Frontend**: React, TypeScript, Axios
- **Deployment**: Google Cloud Run, Docker
- **CI/CD**: Google Cloud Build