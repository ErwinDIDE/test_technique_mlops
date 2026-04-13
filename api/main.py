from fastapi import FastAPI, HTTPException
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List

# Création de l'API
app = FastAPI()

# Chargement de notre modèle
vectorisation_des_phrases = joblib.load("./models/vectorisation.pkl")

@app.get("/")
def home():
    return {"message": "L'API de détection de détection de similarité est en marche"}

# Création de la route API en prenant juste deux phrases
# après on pourra configurer pour accepter une liste de combinaisons de deux phrases
@app.post("/prediction")
def prediction(deux_phrases: List[str]):

    if len(deux_phrases) != 2:
        raise HTTPException(status_code=400, detail="Il faut exactement deux phrases.")

    #on transform les deux phrases
    lesdeux_vecteurs = vectorisation_des_phrases.transform(deux_phrases)

    #on calcule ensuite la similarité cosinus entre le vecteur1 et le vecteur2
    score_similarite = cosine_similarity(lesdeux_vecteurs[0], lesdeux_vecteurs[1])[0][0]

    #on retourne le résutat
    return {
        "phrase_1": deux_phrases[0],
        "phrase_2": deux_phrases[1],
        "score_de_similarite": float(score_similarite)
    }
