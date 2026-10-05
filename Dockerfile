# Eco-Travel Advisor - single container for HuggingFace Spaces (Docker SDK).
# Runs the action server, the Rasa server and the web page; only port 7860 is public.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SQLALCHEMY_SILENCE_UBER_WARNING=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# HuggingFace Spaces runs containers as a non-root user with id 1000
RUN useradd -m -u 1000 user && mkdir /app && chown user:user /app
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=user . .
USER user

# Train during the build so the image contains a ready model
RUN rasa train --fixed-model-name eco-travel

EXPOSE 7860
CMD ["bash", "start.sh"]