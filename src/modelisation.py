import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import wandb
import numpy as np

#initialisation de notre wb
def run_modelisation_maxfeatures_5000():
    wandb.init(
        project="projet_test_technique_mlops_features5000",
        config={
            "model_type": "TF-IDF + Cosine",
            "max_features": 5000,
            "dataset_size": "all"
        }
    )
    df = pd.read_csv("data/processed/dataset_nettoye.csv", sep=';')
    vectorisation_des_phrases = joblib.load("models/vectorisation.pkl")

    vecteurs_1 = vectorisation_des_phrases.transform(df["sentence_1"])
    vecteurs_2 = vectorisation_des_phrases.transform(df["sentence_2"])
    #affichons chaque premier vecteur voir
    print(vecteurs_1[:1])
    print(vecteurs_2[:1])

    similarites = []
    for i in range(len(df)):
        score_de_comparaison = cosine_similarity(vecteurs_1[i],vecteurs_2[i])[0][0]
        similarites.append(score_de_comparaison)

    #on va essayer d'afficher la similarité de notre premiere lige voir ce que ca donne
    print(f"La similarité des deux phrases de la premiere ligne est: {similarites[0]}")
    
    #on sauvegarde le fichier de sortie dans le dossier /results
    df["prediction_score_similarite"] = similarites
    df.to_csv("results/predictions.csv", index=False)
    print("Exportation faite")

    wandb.config.update({
        "nombredelignes_dataset": len(df),
        "colonnes_dataset": list(df.columns)
    })
    #on envoie ensuite les logging de notre model
    wandb.log({
        "moyenne_similarite": np.mean(similarites),
        "similarite_maximale": np.max(similarites)
    })

    #jenregistre le fichier de sortie dans artefact de wangdb
    resultat_artifact = wandb.Artifact(
        name="predictions_maxfeatures_5000",
        type="predictions",
        description="Résultats de la prédiction de la similarité avec un max feature égal à 5000"
    )
    resultat_artifact.add_file("results/predictions.csv")
    wandb.log_artifact(resultat_artifact)   
 
    wandb.finish()

run_modelisation_maxfeatures_5000()

#mon script marche lors de l'éxécution et mon artifact est enregister dans wandb maintenant j'automative la pipeline avec dvc
