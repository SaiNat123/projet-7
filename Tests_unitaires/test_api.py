{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a1c7b012-741b-46c4-812c-eef6d3d6104e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "import joblib\n",
    "import pandas as pd\n",
    "import pytest\n",
    "from flask import Flask, jsonify, request\n",
    "\n",
    "# Ajouter le chemin relatif du fichier api.py au sys.path pour pouvoir l'importer\n",
    "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'API')))\n",
    "\n",
    "# Importer les éléments nécessaires du fichier api.py\n",
    "from api import app, current_directory, model, predict\n",
    "\n",
    "# Créer un client de test pour l'application Flask\n",
    "@pytest.fixture\n",
    "def client():\n",
    "    app.config['TESTING'] = True\n",
    "    with app.test_client() as client:\n",
    "        yield client\n",
    "\n",
    "# Teste le chargement du modèle de prédiction\n",
    "def test_model_loading():\n",
    "    # Détermine le chemin du fichier contenant le modèle entraîné\n",
    "    model_path = os.path.join(current_directory, \"..\", \"Simulations\", \"Best_model\", \"model.pkl\")\n",
    "    # Charge le modèle à partir du fichier\n",
    "    loaded_model = joblib.load(model_path)\n",
    "    # Vérifie que le modèle a été chargé correctement\n",
    "    assert loaded_model is not None, \"Erreur dans le chargement du modèle.\"\n",
    "\n",
    "# Teste le chargement du fichier CSV contenant les données de train\n",
    "def test_csv_loading():\n",
    "    # Détermine le chemin du fichier CSV\n",
    "    csv_path = os.path.join(current_directory, \"..\", \"Simulations\", \"Data\", \"train_final.csv\")\n",
    "    # Charge le fichier CSV dans un DataFrame pandas\n",
    "    df = pd.read_csv(csv_path)\n",
    "    # Vérifie que le DataFrame n'est pas vide\n",
    "    assert not df.empty, \"Erreur dans le chargement du CSV.\"\n",
    "\n",
    "# Teste la fonction de prédiction de l'API\n",
    "def test_prediction(client):\n",
    "    # Détermine le chemin du fichier CSV contenant les données de test\n",
    "    csv_path = os.path.join(current_directory, \"..\", \"Simulations\", \"Data\", \"train_final.csv\")\n",
    "    # Charge le fichier CSV dans un DataFrame pandas\n",
    "    df = pd.read_csv(csv_path)\n",
    "    # Prend un échantillon pour la prédiction\n",
    "    sk_id_curr = df.iloc[0]['SK_ID_CURR']\n",
    "    # Crée une requête de test pour la prédiction en utilisant l'échantillon sélectionné\n",
    "    response = client.post('/predict', json={'SK_ID_CURR': sk_id_curr})\n",
    "    data = response.get_json()  # Utilisation de `get_json` pour obtenir les données JSON de la réponse\n",
    "    prediction = data.get('probability')  # Utilisation de `get` pour éviter une KeyError\n",
    "    # Vérifie que la prédiction a été effectuée correctement\n",
    "    assert prediction is not None, \"La prédiction a échoué.\"\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
