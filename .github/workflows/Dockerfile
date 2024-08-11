# Utiliser une image Python comme base
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier des dépendances
COPY  .github/workflows/requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code
COPY . .

# Exposer le port sur lequel l'API va fonctionner
EXPOSE 5000

# Commande pour démarrer l'API
CMD ["python", "app.py"]
