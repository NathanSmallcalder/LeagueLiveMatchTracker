# League Live Tracker Dockerfile
# Build: docker build -t league-tracker .
# Run: docker run -p 5000:5000 league-tracker

FROM python:3.12-slim

WORKDIR /app

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

CMD ["python", "app.py"]