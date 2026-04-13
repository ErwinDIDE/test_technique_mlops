docker build -t test_technique_mlops .

J'ai inclu le dataset dans le projet et je l'ai versionné avec DVC

#Pour la configuration de WB, j'ai crée un fichier .env pour ma clé API. Il faudra stocker la clé API dans votre fichier
1- Récupérer la clé API sur le site
2- Créer un fichier .env à la racine du projet
3- Ajouter la clé Api en insérant le texte suivant:
	WANDB_API_KEY=le_code_api
4-Lancer la commande bash sur le terminal
	docker run --env-file .env test_technique_mlops
#Jai configuer le fichier .gitignore pour que notre clé ne soit pas visible sur git	
