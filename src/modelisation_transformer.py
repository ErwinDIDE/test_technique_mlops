import os
import pandas as pd
import wandb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_modelisation_transformer():
    # Initialisation de W&B 
    wandb.init(
        project="projet_test_technique_mlops_features5000",
        name="Transformer_MiniLM",
        config={
            "model_type": "Sentence-Transformer",
            "model_name": "all-MiniLM-L6-v2",
            "dataset_size": "all"
        }
    )

    # Chargement des données nettoyées
    df = pd.read_csv("data/processed/dataset_nettoye.csv", sep=';')

    # Chargement du modèle Transformer
    print("Téléchargement/Chargement du modèle Transformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encodage des phrases (Calcul des embeddings sémantiques)
    print("Calcul des embeddings sémantiques ...")
    embeddings_1 = model.encode(df["sentence_1"].tolist(), show_progress_bar=True)
    embeddings_2 = model.encode(df["sentence_2"].tolist(), show_progress_bar=True)

    # 5. Calcul de la similarité cosinus ( avec NumPy)
    print("Calcul des scores de similarité...")
    # Produit scalaire ligne par ligne divisé par la multiplication des normes
    dot_products = np.sum(embeddings_1 * embeddings_2, axis=1)
    norms_1 = np.linalg.norm(embeddings_1, axis=1)
    norms_2 = np.linalg.norm(embeddings_2, axis=1)
    
    norms = norms_1 * norms_2
    similarites = np.where(norms > 0, dot_products / norms, 0.0)

    # Sauvegarde des prédictions
    df["prediction_score_similarite"] = similarites
    df.to_csv("results/predictions_transformer.csv", index=False)
    print("Exportation faite dans results/predictions_transformer.csv")

    # ÉVALUATION
    vrai_score = df["similarity_score"] 
    mse = mean_squared_error(vrai_score, similarites)
    mae = mean_absolute_error(vrai_score, similarites)
    
    print(f"📊 ÉVALUATION TRANSFORMER -- MSE: {mse:.4f} | MAE: {mae:.4f}")

    # Mise à jour de la config W&B
    wandb.config.update({
        "nombredelignes_dataset": len(df),
        "embedding_dim": embeddings_1.shape[1]
    })
    
    # Envoi des métriques dans W&B
    wandb.log({
        "eval/MSE": mse,
        "eval/MAE": mae,
        "stats/moyenne_similarite": np.mean(similarites),
        "stats/similarite_maximale": np.max(similarites)
    })

    # Enregistrement de l'artéfact
    resultat_artifact = wandb.Artifact(
        name="predictions_transformer",
        type="predictions",
        description="Résultats de la prédiction de la similarité avec Sentence-Transformer"
    )
    resultat_artifact.add_file("results/predictions_transformer.csv")
    wandb.log_artifact(resultat_artifact)   
 
    wandb.finish()

if __name__ == "__main__":
    run_modelisation_transformer()
