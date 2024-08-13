import os
import joblib
import pandas as pd
import shap
from flask import Flask, jsonify, request

app = Flask(__name__)

# Récupérez le répertoire actuel du fichier api.py
current_directory = os.path.dirname(os.path.abspath(__file__))

# Charger le modèle et le scaler
model_path = os.path.join(current_directory, "Simulations", "Best_model", "model.pkl")
model = joblib.load(model_path)

scaler_path = os.path.join(current_directory, "Simulations", "Scaler", "StandardScaler.pkl")
scaler = joblib.load(scaler_path)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Obtenir les données de la requête
        data = request.json
        sk_id_curr = data.get('SK_ID_CURR')
        
        if sk_id_curr is None:
            return jsonify({'error': 'SK_ID_CURR is required'}), 400

        # Construire le chemin complet vers df_train.csv
        csv_path = os.path.join(current_directory, "Simulations", "Data", "df_train.csv")
        df = pd.read_csv(csv_path)

        # Préparer l'échantillon pour la prédiction
        sample = df[df['SK_ID_CURR'] == sk_id_curr]

        if sample.empty:
            return jsonify({'error': 'Client ID not found'}), 404

        # Supprimer les colonnes non nécessaires pour la prédiction
        sample = sample.drop(columns=['SK_ID_CURR', 'TARGET'])

        # Appliquer le scaler
        sample_scaled = scaler.transform(sample)

        # Prédire
        prediction = model.predict_proba(sample_scaled)
        proba = prediction[0][1]# Probabilité de la seconde classe
        
        # Calculer les valeurs SHAP pour l'échantillon donné
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_scaled)

        # Retourner les valeurs SHAP avec la probabilité
        return jsonify({
            'probability': proba * 100,
            'shap_values':  ' ', #shap_values[1][0].tolist(),
            'feature_names': sample.columns.tolist(),
            'feature_values': sample.values[0].tolist()
        })

    except Exception as e:
        # Log the error and return a JSON response
        app.logger.error(f'Error: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=True, host="0.0.0.0", port=int(port))
