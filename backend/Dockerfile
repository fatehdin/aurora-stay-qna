
# Minimal Python base image
FROM python:3.13.15-slim

# Working directory inside the container
WORKDIR /app
# Copy the app, rag_utils.py and the saved FAISS index into the image
COPY . .

# Upgrade pip, then install the CPU-only build of PyTorch (smaller image, no GPU needed)
RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install the remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Folder where the downloaded embedding model is cached inside the image
ENV HF_HOME=/app/huggingface

# Download the model at build time
RUN python -c "\
from langchain_huggingface import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings( \
    model_name='HarishMaths/Hotel-Policy-Embedding', \
    model_kwargs={'device': 'cpu'}, \
    encode_kwargs={'normalize_embeddings': True} \
)"

# Force offline mode at runtime so it uses the cached model
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# Start the Flask app with gunicorn on port 7860 (the generous timeout covers the LLM call)
CMD ["gunicorn", "-w", "1", "--timeout", "120", "-b", "0.0.0.0:7860", "app:rag_api"]
