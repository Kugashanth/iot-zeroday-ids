FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only PyTorch keeps the image small
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[app]"

COPY configs ./configs
COPY app ./app

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# default: build demo data + results if none exist, then serve the dashboard
CMD ["sh", "-c", "[ -d runs ] || (iotids synth --out data/synthetic && iotids all --config configs/synthetic.yaml); streamlit run app/streamlit_app.py --server.address=0.0.0.0"]
