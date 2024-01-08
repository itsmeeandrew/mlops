# Base image
FROM python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt project/requirements.txt
COPY pyproject.toml project/pyproject.toml
COPY mlops/ project/mlops/
COPY data/ project/data/
COPY reports/ project/reports/

WORKDIR /project
RUN pip install . --no-cache-dir


ENTRYPOINT ["python", "-u", "mlops/predict_model.py"]
