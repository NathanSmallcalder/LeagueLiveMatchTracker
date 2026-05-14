FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY df_wide.csv scaler_*.pkl lol_model_*.pth ./
COPY templates/ ./templates/

ENV PYTHONUNBUFFERED=1
ENV RIOT_API_URL=http://host.docker.internal:2999/liveclientdata/allgamedata

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
