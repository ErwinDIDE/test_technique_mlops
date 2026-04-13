import pandas as pd
import re

df = pd.read_csv("data/raw/dataset.csv", sep=";")

#On a exploré notre dataset visuellement. On va s'assurer qu'aucune ligne ne tient de valeures vides meme si visuellement il n'en contient pas
#On va d'abord supprimer une ligne si par exemple une colonne sentence1, sentence2 ou similaryscore est vide pour éviter les erreurs dans l'entrainement du model
df = df.dropna(subset=['sentence_1', 'sentence_2', 'similarity_score'])

#On supprime ensuite les lignes dupliquées éventuellement si il y en a
df = df.drop_duplicates()

#Lorsque j'ai visualise le dataset il y a des crochets et des guillemets. Il faut les supprimer pour l'approche tf-idf
def nettoyage_phrase(phrase):
    #on va supprimer les crochets et les parentheses et garder les textes uniquement
    phrase = re.sub(r"[\[\]\(\)]", " ", phrase)
    #supprimer les doubles espaces crés
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase.lower()

#je l'applique sur les deux colonnes et ensuite j enregistre mon fichier de sortie
df["sentence_1"] = df["sentence_1"].apply(nettoyage_phrase)
df["sentence_2"] = df["sentence_2"].apply(nettoyage_phrase)

#On enregistre la sortie dans notre dossier /processed qu'on a crée précedemment
df.to_csv("data/processed/dataset_nettoye.csv", index=False, sep=";") #garder le meme format de fichier csv pour nos deux datasets

print("Apercu du dataset /n")
df.info()

print("Nettoyage effectué avec succès")


