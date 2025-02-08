FROM python:3.10.8-slim

WORKDIR /app
COPY requirements-fe.txt .
RUN pip install -r requirements-fe.txt

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

COPY gradio_launch.py .

CMD ["python", "gradio_launch.py"]