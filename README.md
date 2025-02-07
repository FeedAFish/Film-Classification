# Film Classification and Recommendation System

This project provides a web interface for film poster classification and recommendation using deep learning.

Film classification can be replaced with other dataset and classification (animal, etc)

## Features
- Image classification of film posters
- Similar image recommendations
- User-friendly web interface
- Containerized deployment

## Prerequisites
- Docker and Docker Compose
- Python 3.9+
- CUDA-compatible GPU (optional)

## Installation and Run Local

Clone the repository:
```bash
git clone https://github.com/yourusername/Film-Classification.git
cd Film-Classification
```

1. Python

```bash
pip install -r requirements-be.txt -r requirements-fe.txt
python ./api_server.py & ./gradio_launch.py
```

2. Docker
```bash
docker-compose up --build
```
## Usage

1. Access the web interface:
   - Local: http://localhost:7860

2. Upload an image:
   - Click the upload area or drag & drop a film poster
   - The system will classify the image and show similar recommendations

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

## API Endpoints

- `/predict`: Classification endpoint
- `/recommend`: Recommendation endpoint

## Environment Variables

- `API_URL`: Backend API URL (default: http://localhost:8000)

