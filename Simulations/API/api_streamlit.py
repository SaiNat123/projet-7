import streamlit as st
import requests

# Définir l'URL de l'API Flask
API_BASE_URL = "http://localhost:5000"

# Fonction pour obtenir la prédiction d'un client
def get_prediction(SK_ID_CURR):
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json={'SK_ID_CURR': SK_ID_CURR})
        if response.ok:
            return response.json()
        else:
            # Afficher l'erreur retournée par l'API
            error_message = response.json().get('error', 'Erreur inconnue')
            st.error(f"Erreur lors de la récupération des prédictions : {error_message}")
            return {}
    except requests.RequestException as e:
        st.error(f"Erreur lors de la connexion à l'API : {e}")
        return {}

# Application Streamlit
st.title("Analyse de Prédiction de Défaut")

# Saisie manuelle du numéro de client
client_id_input = st.text_input("Entrez le numéro de client:")

if client_id_input:
    try:
        client_id = int(client_id_input)
    except ValueError:
        st.error("Veuillez entrer un numéro de client valide.")
    else:
        if st.button("Obtenir Prédiction"):
            prediction = get_prediction(client_id)
            if prediction:
                probability = prediction.get('probability', 0)
                st.write(f"Probabilité de défaut : {probability:.2f}%")
                
                # Afficher le message basé sur la probabilité
                if probability < 40:
                    st.success("Le prêt sera accordé.")  # Texte en vert
                else:
                    st.error("Le prêt ne sera pas accordé.")  # Texte en rouge
else:
    st.write("Veuillez entrer un numéro de client pour obtenir la prédiction.")
