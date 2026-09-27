FROM python:3.12-slim

# /app/backend, not /app: src/config.py finds models_store/ and .env from the repo root,
# two levels above src/.
WORKDIR /app/backend

COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/prod.txt

# Bake the encoder into the image so a cold start does not download it from the HF Hub.
# Name, revision and files come from EMBEDDING_MODEL_* in src/ml/constants.py (which imports
# src/config.py; its required DATABASE_URL gets a placeholder, nothing connects here). Only
# those two files are copied, so the download layer stays cached across other code changes.
COPY src/config.py src/config.py
COPY src/ml/constants.py src/ml/constants.py
RUN DATABASE_URL=unused python -c "from huggingface_hub import hf_hub_download as d; from src.ml.constants import EMBEDDING_MODEL_NAME as n, EMBEDDING_MODEL_REVISION as r, EMBEDDING_MODEL_FILES as fs; [d(n, f, revision=r) for f in fs]"
# Load it from that cache at run time without asking the Hub for updates.
ENV HF_HUB_OFFLINE=1
# numpy's OpenBLAS starts one thread per visible host core at import, whatever the CPU quota.
# The app does no large BLAS work (onnxruntime and the tokenizer have their own pools, capped
# in src/ml/aspects.py), so one thread is enough.
ENV OPENBLAS_NUM_THREADS=1

COPY . .

EXPOSE 8000
# Render sets PORT; locally it defaults to 8000.
CMD ["sh", "-c", "alembic upgrade head && uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-config logging.ini"]
