ETAPE 2
docker build -t test_technique_mlops .

ETAPE 3
#Lorsque je faisais un simple docker run mes models et les resultats n'était pas sauvegarder et mes dossiers models, results et /data/processed étaient vides
#Cela est du au fait que le container docker les envoie dans les dossiers /app/models, ..... qu'on a défini dans notre Dockerfile
#Pour palier ca il faut rajouter les parametres -v en éxécutant la commande suivante:

docker run -e WANDB_API_KEY=<VOTRE_CLE_API_WANDB_ICI_SANS_LES_BALISES> \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/data:/app/data" \
  test_technique_mlops

J'ai inclu le dataset dans le projet et je l'ai versionné avec DVC

ETAPE 1
#Pour la configuration de WB, j'ai crée un fichier .env pour ma clé API. Il faudra stocker la clé API dans votre fichier
1- Récupérer la clé API sur le site
2- Créer un fichier .env à la racine du projet
3- Ajouter la clé Api en insérant le texte suivant:
	WANDB_API_KEY=le_code_api
4-Lancer la commande bash sur le terminal pour injecter la variable dans le docker
	docker run --env-file .env test_technique_mlops
#Jai configuer le fichier .gitignore pour que notre clé ne soit pas visible sur git	

ETAPE 5: TEST DE L'API
taper la commande suivante:
uvicorn api.main:app --reload
