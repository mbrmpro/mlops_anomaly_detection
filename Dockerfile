FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY src/ src/

# Only .git/HEAD and .git/refs are in the build context (see .dockerignore).
# src/training.py reads them to tag each MLflow run with the image's commit.
COPY .git/ .git/

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
