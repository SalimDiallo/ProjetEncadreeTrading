# =============================================================
# Dockerfile — Petrol Trading Platform
# =============================================================
# Image basée sur Python 3.12-slim (~150 MB de base)
#
# Note technique : on n'utilise pas Python 3.14 (requis par
# pyproject.toml) car il est en pré-release et plusieurs libs
# ML/data (XGBoost, scikit-learn) ne le supportent pas encore.
# Python 3.12 est stable et fonctionne avec toutes les libs.
# =============================================================

FROM python:3.12-slim

LABEL maintainer="Projet Encadré INSEA S4"
LABEL description="Plateforme de backtesting algorithmique — WTI Crude Oil"

# -----------------------------------------------------------
# Variables d'environnement
# -----------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONPATH=/workspace

# -----------------------------------------------------------
# Dépendances système
# -----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------
# Répertoire de travail
# -----------------------------------------------------------
WORKDIR /workspace

# -----------------------------------------------------------
# Installation des dépendances Python
# Note : on copie d'abord uniquement requirements.txt pour
# bénéficier du cache Docker. Si le code change mais pas les
# dépendances, on évite de tout réinstaller.
# -----------------------------------------------------------
COPY web/requirements.txt /tmp/web_requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/web_requirements.txt && \
    pip install \
        "scikit-learn>=1.4.0" \
        "xgboost>=2.0.0" \
        "matplotlib>=3.8.0" \
        "seaborn>=0.13.0" \
        "requests>=2.31.0" \
        "beautifulsoup4>=4.12.0" \
        "lxml>=5.0.0" \
        "jupyter>=1.0.0" \
        "ipykernel>=6.28.0" \
        "notebook>=7.0.0" \
        "jupyterlab>=4.0.0"

# -----------------------------------------------------------
# Copie du code source
# (sera écrasé par le volume monté en mode dev container)
# -----------------------------------------------------------
COPY . /workspace/

# -----------------------------------------------------------
# Healthcheck pour Docker Compose
# -----------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# -----------------------------------------------------------
# Ports exposés
# 8501 = Streamlit Dashboard
# 8888 = Jupyter Lab
# -----------------------------------------------------------
EXPOSE 8501 8888

# -----------------------------------------------------------
# Commande par défaut : lancer Streamlit
# Peut être overridée par docker-compose.yml ou docker run
# -----------------------------------------------------------
CMD ["streamlit", "run", "web/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
