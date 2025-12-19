# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# CONFIGURATION DE LA PAGE
# ---------------------------
st.set_page_config(page_title="Inclusion Financière - Prédiction", page_icon="💰")
st.title("💰 Prédiction de l'inclusion financière")
st.markdown("Entrez les informations du répondant pour prédire s'il possède un compte bancaire.")

# ---------------------------
# CHARGEMENT DU MODELE ET ENCODEURS
# ---------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("financial_inclusion_model.pkl")      # modèle entraîné
        encoders = joblib.load("label_encoders (2).pkl")              # encodeurs pour les catégorielles
        return model, encoders
    except FileNotFoundError as e:
        st.error(f"❌ Fichier manquant : {e}")
        st.stop()

model, label_encoders = load_model()

# ---------------------------
# SAISIE DES DONNÉES
# ---------------------------
st.header("Informations du répondant")

country = st.selectbox("Pays", options=label_encoders['country'].classes_)
year = st.number_input("Année", min_value=2000, max_value=2030, value=2025)
location_type = st.selectbox("Type de localisation", options=label_encoders['location_type'].classes_)
cellphone_access = st.selectbox("Accès au téléphone portable ?", options=label_encoders['cellphone_access'].classes_)
household_size = st.number_input("Taille du foyer", min_value=1, value=3)
age = st.number_input("Âge du répondant", min_value=0, max_value=120, value=30)
gender = st.selectbox("Genre", options=label_encoders['gender_of_respondent'].classes_)
relationship = st.selectbox("Relation avec le chef de ménage", options=label_encoders['relationship_with_head'].classes_)
marital_status = st.selectbox("Statut marital", options=label_encoders['marital_status'].classes_)
education = st.selectbox("Niveau d'éducation", options=label_encoders['education_level'].classes_)
job_type = st.selectbox("Type d'emploi", options=label_encoders['job_type'].classes_)

# ---------------------------
# PRÉPARATION DES DONNÉES
# ---------------------------
input_dict = {
    "country": country,
    "year": year,
    "location_type": location_type,
    "cellphone_access": cellphone_access,
    "household_size": household_size,
    "age_of_respondent": age,
    "gender_of_respondent": gender,
    "relationship_with_head": relationship,
    "marital_status": marital_status,
    "education_level": education,
    "job_type": job_type
}

# Encodage
for col in input_dict:
    if col in label_encoders:
        input_dict[col] = label_encoders[col].transform([input_dict[col]])[0]

# DataFrame pour la prédiction
input_df = pd.DataFrame([input_dict])

# ---------------------------
# PRÉDICTION
# ---------------------------
if st.button("Prédire l'inclusion financière"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.success(f"✅ Le répondant est susceptible de posséder un compte bancaire (Probabilité: {proba[1]:.2%})")
    else:
        st.warning(f"❌ Le répondant est peu susceptible de posséder un compte bancaire (Probabilité: {proba[0]:.2%})")
