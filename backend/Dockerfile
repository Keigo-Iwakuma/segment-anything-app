FROM nvcr.io/nvidia/pytorch:23.01-py3

WORKDIR /usr/src/app

RUN pip install fastapi[all]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]