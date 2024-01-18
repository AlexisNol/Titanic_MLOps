FROM python:3.8

RUN apt update -y && apt install awscli -y

# Répertoire de travail
WORKDIR /app

# Copie du répertoire
COPY . /app

# Installation des dépendances
RUN pip install -r requirements.txt

# Commande pour démarrer l'application
CMD ["python3", "app.py"]
