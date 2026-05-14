# League Live Tracker Dockerfile
# Build: docker build -t league-tracker .
# Run: docker run -p 5000:5000 league-tracker

FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir flask flask-limiter requests urllib3 joblib pandas numpy gunicorn

# Copy application files
COPY app.py ./
COPY df_wide.csv scaler_*.pkl lol_model_*.pth ./

# Copy templates
COPY templates/ ./templates/

# Configure environment
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV RIOT_API_URL=http://host.docker.internal:2999/liveclientdata/allgamedata

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4"]