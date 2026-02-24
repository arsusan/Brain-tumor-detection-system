# Use Python 3.11-slim for better compatibility with TensorFlow 2.15.0
FROM python:3.11-slim

# 1. Set environment variables early to keep the build clean
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app

# 2. Setup non-root user for security (Hugging Face requirement)
RUN useradd -m -u 1000 user
WORKDIR /app

# 3. Install system dependencies
# Combined into one layer and cleaned up to keep image size small
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python dependencies
# We copy only requirements.txt first to leverage Docker layer caching
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application files
# We do this AFTER installing requirements so code changes don't trigger re-installation of libraries
COPY --chown=user:user . .


RUN mkdir -p /home/user/.cache && chmod -R 777 /home/user/.cache
RUN chmod -R 777 /app
# 6. Switch to non-root user
USER user

# 7. Expose the Hugging Face default port
EXPOSE 7860

# 8. Start the application
# We point to backend.main:app because your code is inside the /backend folder
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]