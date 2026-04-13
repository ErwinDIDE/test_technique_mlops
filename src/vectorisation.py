import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/processed/dataset_nettoye.csv", sep=';')

vectorisation_des_phrases = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
phrases_du_dataset = pd.concat([df["sentence_1"],df["sentence_2"]])
print(phrases_du_dataset[:3])

vectorisation_des_phrases.fit(phrases_du_dataset)

noms_de_vectorisation = vectorisation_des_phrases.get_feature_names_out()
print(noms_de_vectorisation[:10])

#On observe les termes les plus important abrasive cloth , ....
#et les moins importants sont:
print(noms_de_vectorisation[-10:])

#on va sauvegarder notre model dans /models
joblib.dump(vectorisation_des_phrases, "models/vectorisation.pkl")
print("La vectorisation a été éffectué avec succès")
