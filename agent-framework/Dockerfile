FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONPATH=/app/src

# Run the ARQ worker in the background and keep uvicorn as the main process.
CMD ["sh", "-c", "arq finflow_agent.api.WorkerSettings & exec uvicorn finflow_agent.api:app --host 0.0.0.0 --port 8001"]
