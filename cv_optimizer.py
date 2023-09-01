import openai
import streamlit as st
import gensim.downloader as api
from gensim.utils import simple_preprocess
import PyPDF2
import tempfile
import os
import numpy as np

# Charger le modèle glove-wiki pré-entraîné
model = api.load('glove-wiki-gigaword-50')

# Configurer votre clé API OpenAI
openai.api_key = "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"

# Interface utilisateur Streamlit
uploaded_file = st.file_uploader("Importer votre CV", type=["pdf"])
job_offer_text = st.text_area("Coller le texte de l'offre d'emploi ici", height=300, max_chars=6000)

if uploaded_file is not None and job_offer_text:
    # Enregistrer le fichier téléchargé temporairement
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Extraire le texte du PDF en utilisant PyPDF2
    cv_text = ""
    with open(temp_file_path, 'rb') as pdf_file:
        pdf = PyPDF2.PdfReader(pdf_file)
        for page in pdf.pages:
            cv_text += page.extract_text()

    # Supprimer le fichier temporaire
    os.remove(temp_file_path)

    # Obtenir les embeddings pour les mots du texte en ignorant les mots qui ne sont pas dans le vocabulaire du modèle
    try:
        job_offer_embedding = np.mean([model[word] for word in simple_preprocess(job_offer_text) if word in model], axis=0)
    except KeyError:
        job_offer_embedding = np.zeros(model.vector_size)
    try:
        cv_embedding = np.mean([model[word] for word in simple_preprocess(cv_text) if word in model], axis=0)
    except KeyError:
        cv_embedding = np.zeros(model.vector_size)

    # Calculer la similarité entre les embeddings
    similarity = np.dot(job_offer_embedding, cv_embedding) / (np.linalg.norm(job_offer_embedding) * np.linalg.norm(cv_embedding))

    # Utiliser l'API OpenAI pour générer des suggestions
    prompt = f"Offre d'emploi : {job_offer_text}\nCV : {cv_text}\nSimilarité : {similarity}\nSuggestions :"
    response = openai.Completion.create(
        engine="text-davinci-003",  # Choisissez le moteur de génération
        prompt=prompt,
        temperature=0.6,             # Contrôlez la créativité des réponses
        max_tokens=100               # Limitez le nombre de tokens à 100
    )

    # Recuperer et Afficher la suggestion générée dans Streamlit
    suggestions = response.choices[0].text.strip().split("###SEPARATEUR###")
    st.write("Suggestion générée pour adapter le CV à l'offre d'emploi:")
    for suggestion in suggestions:
        st.write(suggestion)
