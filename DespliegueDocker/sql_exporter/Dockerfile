FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY sql_avg_pusher.py .
CMD ["python", "sql_avg_pusher.py"]
