# ==============================================================================
# RecipeGen AI — Production Container Image
# Base Image: Official Python 3.12 Slim (Debian Bookworm)
# ==============================================================================

FROM python:3.12-slim

# Maintainer / Metadata labels
LABEL maintainer="RecipeGen AI Team"
LABEL description="Ingredient-Conditioned Recipe Generation & Safety Platform"

# Prevent interactive prompts during package installation and tune runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    KERAS_BACKEND=tensorflow \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8501 \
    MALLOC_TRIM_THRESHOLD_=100000

# Install lightweight OS dependencies (curl for healthcheck, ca-certificates)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user and directory structure
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m -s /bin/bash appuser && \
    mkdir -p /app /app/models /app/data /tmp/.keras && \
    chown -R appuser:appgroup /app /tmp/.keras

WORKDIR /app

# Step 1: Install Python dependencies from requirements.txt to system site-packages
COPY --chown=appuser:appgroup requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Step 2: Copy application source code and configuration
COPY --chown=appuser:appgroup app/ /app/app/
COPY --chown=appuser:appgroup utils/ /app/utils/
COPY --chown=appuser:appgroup .streamlit/ /app/.streamlit/

# Step 3: Copy processed evaluation outputs & datasets (required by Training & Analysis pages)
COPY --chown=appuser:appgroup data/outputs/ /app/data/outputs/
COPY --chown=appuser:appgroup data/processed/ /app/data/processed/

# Step 4: Copy Stage 2F production model weights & configuration
COPY --chown=appuser:appgroup models/recipe_gpt2_full/ /app/models/recipe_gpt2_full/

# Switch to non-root user
USER appuser

# Expose default Streamlit port (configurable at runtime via PORT environment variable)
EXPOSE 8501

# Container Healthcheck verifying Streamlit HTTP response
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Start Streamlit application
CMD ["sh", "-c", "streamlit run app/main.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
