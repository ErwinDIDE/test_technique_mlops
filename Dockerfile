FROM python:3.11-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p models results

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

#ensuite ignorer git avec dvc
RUN dvc init --no-scm -f

CMD ["dvc", "repro"]


