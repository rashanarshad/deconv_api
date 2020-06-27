FROM python:3.7

COPY ./app /app
# COPY requirements.txt /app
WORKDIR /app

#requirements.txt /app ./mrcnn /app
RUN pip install -r requirements.txt

EXPOSE 80




CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
