FROM python:3.9-slim

WORKDIR /app
COPY requirements-be.txt .
RUN pip install -r requirements-be.txt

COPY api_server.py .
COPY utils/ ./utils/
COPY data/ ./data/
COPY model.mdl .

CMD ["python", "api_server.py"]