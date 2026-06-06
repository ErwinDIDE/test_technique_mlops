import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import wandb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    
    # CALCUL OPTIMISÉ (sans la boucle for)
    # Calcule la similarité cosinus ligne par ligne de manière matricielle
    # cos_sim = (A . B) / (||A|| * ||B||)
    dot_products = np.array(vecteurs_1.multiply(vecteurs_2).sum(axis=1)).flatten()
    norm_1 = np.array(np.sqrt(vecteurs_1.multiply(vecteurs_1).sum(axis=1))).flatten()
    norm_2 = np.array(np.sqrt(vecteurs_2.multiply(vecteurs_2).sum(axis=1))).flatten()
    
    # Gestion de la division par zéro au cas où une phrase n'aurait aucun mot du vocabulaire
    norms = norm_1 * norm_2
    similarites = np.where(norms > 0, dot_products / norms, 0.0)

    print(f"La similarité des deux phrases de la première ligne est: {similarites[0]}")
    
    # Sauvegarde du fichier de sortie dans le dossier /results
    df["prediction_score_similarite"] = similarites
    df.to_csv("results/predictions.csv", index=False)
    print("Exportation faite dans results/predictions.csv")

    # 4. CALCUL DES VRAIES MÉTRIQUES DE RÉGRESSION (MAE et MSE)
    # On suppose que la vraie colonne s'appelle 'similarity_score' (à adapter selon ton CSV)
    vrai_score = df["similarity_score"] 
    
    mse = mean_squared_error(vrai_score, similarites)
    mae = mean_absolute_error(vrai_score, similarites)
    
    print(f"📊 ÉVALUATION -- MSE: {mse:.4f} | MAE: {mae:.4f}")

    # Mise a jour de la config W&B
    wandb.config.update({
        "nombredelignes_dataset": len(df),
        "colonnes_dataset": list(df.columns)
    })
    
    # Envoie des métriques dans W&B
    wandb.log({
        "eval/MSE": mse,
        "eval/MAE": mae,
        "stats/moyenne_similarite": np.mean(similarites),
        "stats/similarite_maximale": np.max(similarites)
    })

    # 6. Enregistrement du fichier de sortie dans les artefacts de W&B
    resultat_artifact = wandb.Artifact(
        name="predictions_maxfeatures_5000",
        type="predictions",
        description="Résultats de la prédiction de la similarité avec un max feature égal à 5000"
    )
    resultat_artifact.add_file("results/predictions.csv")
    wandb.log_artifact(resultat_artifact)   
 
    wandb.finish()

if __name__ == "__main__":
    run_modelisation_maxfeatures_5000()
