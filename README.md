ETAPE 2
docker build -t test_technique_mlops .

ETAPE 3

docker run -e WANDB_API_KEY=<CLE_API_SANS_BALISES> -v "$(pwd)/models:/app/models" -v "$(pwd)/results:/app/results" -v "$(pwd)/data:/app/data" test_technique_mlops
Dataset inclut dans le projet et versionné avec DVC

ETAPE 1
#Pour la configuration de WB, un fichier .env a été crée pour la clé API. Il faudra stocker la clé API dans votre fichier
1- Récupérer la clé API sur le site
2- Créer un fichier .env à la racine du projet
3- Ajouter la clé Api en insérant le texte suivant:
	WANDB_API_KEY=le_code_api
4-Lancer la commande bash sur le terminal pour injecter la variable dans le docker
	docker run --env-file .env test_technique_mlops
#Fichier .gitignore configuré  pour que la clé ne soit pas visible sur git	

ETAPE 5: TEST DE L'API
taper la commande suivante:
uvicorn api.main:app --reload
