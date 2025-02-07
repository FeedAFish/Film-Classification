# Film Classification and Recommendation System

This project provides a web interface for film poster classification and recommendation using deep learning.

> Note: This system can be adapted for other image classification tasks (e.g., animals, objects, etc.)

## Features
- Image classification of film posters
- Similar image recommendations
- User-friendly web interface
- Containerized deployment

## Prerequisites
- Docker and Docker Compose
- Python 3.9+
- CUDA-compatible GPU (optional)

## Installation and Setup

### 1. Local Python Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/Film-Classification.git
cd Film-Classification

# Install dependencies
pip install -r requirements-be.txt -r requirements-fe.txt

# Run the application
# For Windows
start python api_server.py & start python gradio_launch.py

# For Linux
python api_server.py & python gradio_launch.py &
```

### 2. Docker Setup
```bash
docker-compose up --build
```

## Usage

1. Access the web interface at http://localhost:7860

2. Upload an image:
   - Click the upload area or drag & drop a film poster
   - Wait for classification results and similar recommendations

## Project Structure
```
Film-Classification/
├── utils/                # Utility functions
├── api_server.py         # FastAPI backend server
├── gradio_launch.py      # Gradio frontend interface
├── docker-compose.yml    # Docker composition file
├── frontend.Dockerfile   # Frontend container configuration
├── backend.Dockerfile    # Backend container configuration
└── .gitignore
```

## API Documentation

### Endpoints
- `/predict`: Classification endpoint
- `/recommend`: Recommendation endpoint

### Environment Variables
- `API_URL`: Backend API URL (default: http://localhost:8000)

## Limitations and Notes

- Currently works with local `sqlite3` database only
- For production deployment with different databases, modify `load_kdtree` method in `utils/dataloader.py`
- Model files and data are not included in the repository