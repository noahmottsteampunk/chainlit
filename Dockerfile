FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir chainlit openai mcp plotly pandas

COPY azure_openai_demo.py azure_openai_mcp_demo.py ./
COPY .chainlit/ ./.chainlit/
COPY public/ ./public/

ENV AZURE_OPENAI_API_KEY=""
ENV CHAINLIT_APP_ROOT=/app

EXPOSE 8000

CMD ["chainlit", "run", "azure_openai_mcp_demo.py", "--host", "0.0.0.0", "-w"]
