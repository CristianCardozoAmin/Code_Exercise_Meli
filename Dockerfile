FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./src ./src
COPY ./data/models ./data/models

EXPOSE 8000

CMD ["uvicorn", "src.api_model_serving:app", "--host", "0.0.0.0", "--port", "8000"]