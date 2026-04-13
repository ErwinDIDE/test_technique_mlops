FROM python:3.11-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p models results

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN dvc init --no-scm -f

CMD ["dvc", "repro"]

#Pour la production kubernetes il faudra que je creer plusieurs Dockerfile pour construire chaque image(entrainement, modelisation, api) et les mettre sur le Docker hub

