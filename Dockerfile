FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models

VOLUME /app/input
VOLUME /app/output

CMD ["python", "./app/app.py"]