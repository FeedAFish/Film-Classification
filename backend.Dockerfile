FROM python:3.10.8-slim

WORKDIR /app
COPY requirements-be.txt .
RUN pip install -r requirements-be.txt

COPY api_server.py .
COPY demo_dl.py .
COPY utils/ ./utils/

RUN python demo_dl.py

ENV DOCKER=true

CMD ["python", "api_server.py"]