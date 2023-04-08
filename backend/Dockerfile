FROM nvcr.io/nvidia/pytorch:23.01-py3

WORKDIR /app

COPY requirement.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
