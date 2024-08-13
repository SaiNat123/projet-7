# Utiliser une image Python comme base
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier des dépendances
COPY .github/workflows/requirements.txt /app/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code, y compris les tests
COPY . /app/

# Exposer le port sur lequel l'API va fonctionner
EXPOSE 5000

# Définir la commande par défaut
CMD ["python", "app.py"]

