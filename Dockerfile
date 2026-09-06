FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY dashboard ./dashboard
COPY conftest.py pytest.ini ./

RUN mkdir -p data/sample data/raw data/processed

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]