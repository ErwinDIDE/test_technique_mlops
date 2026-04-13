FROM python:3.11-slim

WORKDIR /app

RUN mkdir -p models results

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["dvc", "repro"]

#Pour la production kubernetes il faudra que je creer plusieurs Dockerfile pour construire chaque image(entrainement, modelisation, api) et les mettre sur le Docker hub
