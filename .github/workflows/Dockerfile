# Utiliser une image Python légère comme base
FROM python:3.10-slim
# Installer libgomp1 pour résoudre les dépendances de parallélisme
RUN apt-get update && apt-get install -y libgomp1 && apt-get clean

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier des dépendances
COPY .github/workflows/requirements.txt /app/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le répertoire contenant l'API
COPY ../Simulations/API /app/

# Copier les données de test
COPY ../Simulations/Data/df_train.csv /app/df_train.csv

# Copier le modèle
COPY ../Simulations/Best_model /app/

# Copier le scaler
COPY ../Simulations/Scaler /app/

# Copier le fichier de test unitaire
COPY ../Tests_unitaires/test_api.py /app/test_api.py

# Exposer le port sur lequel l'API va fonctionner
EXPOSE 5000

# Commande par défaut pour démarrer l'application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api:app"]
