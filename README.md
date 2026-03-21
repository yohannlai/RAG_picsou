18/03/2026 **LAI Yohann - E4 IMAC**

# 💰 TP RAG — Le Coffre de Picsou 

Bienvenue dans le coffre-fort le plus sécurisé de Donaldville ! 🦆

Ce projet implémente un système RAG (Retrieval-Augmented Generation) en Python. Il permet de poser des questions sur l'univers de Balthazar Picsou en se basant strictement sur un corpus de textes extraits du Picsou Wiki de Fandom.

L'assistant IA est configuré pour faire du **roleplay** : il répond à la première personne avec le caractère avare, grognon mais amusant de l'oncle Picsou, tout en s'assurant de ne jamais halluciner d'informations hors du corpus.

🌐 **Tester l'application en ligne :** [Le Coffre de Picsou](https://le-coffre-de-picsou.onrender.com) *(Hébergé sur Render)*

---

## 🛠️ Stack Technique

* **Orchestration :** LangChain
* **Base de données vectorielle :** FAISS (Facebook AI Similarity Search)
* **Modèle d'Embeddings :** `all-MiniLM-L6-v2` (Architecture hybride : Local via CPU / Serveur via API HuggingFace pour économiser la RAM)
* **LLM (Génération) :** `arcee-ai/trinity-large-preview:free` (via l'API OpenRouter, optimisé pour le roleplay)
* **Traitement du texte :** Découpage optimisé (chunks de 1500 caractères avec un chevauchement de 500 caractères).
* **Interface Web :** Flask, HTML/CSS/JS natif.

---

## 📂 Architecture du Projet

Le dépôt contient les fichiers suivants :

~~~text
RAG_picsou/
├── corpus/
│   └── picsou/               # Dossier contenant les 39 fichiers .txt du Wiki
├── faiss_index/              # Base vectorielle pré-calculée (Optimisation pour Render)
├── templates/
│   └── index.html            # Interface visuelle web stylisée "Coffre-Fort"
├── .gitignore                # Fichier ignorant les données sensibles (.env, venv, etc.)
├── README.md                 # Documentation du projet
├── requirements.txt          # Dépendances complètes pour l'exécution locale (avec Torch)
├── requirements_render.txt   # Dépendances allégées pour le déploiement serveur (sans Torch)
├── main.py                   # Script principal du RAG (version terminal interactive)
├── main_flask.py             # Serveur Python gérant l'application web
└── wiki_downloader.py        # Script utilitaire pour télécharger les pages du Wiki Fandom
~~~

---

## 🚀 Installation et Lancement (Mode Local)

Si vous souhaitez faire tourner mon coffre-fort sur votre propre machine, suivez ces instructions à la lettre. Le temps, c'est de l'argent !

### 1. Prérequis
Assurez-vous d'avoir Python 3 installé sur votre machine.

### 2. Cloner le dépôt et préparer l'environnement
Clonez ce projet sur votre machine, puis créez et activez un environnement virtuel pour isoler les dépendances :

~~~bash
# Clonage du repo
git clone https://github.com/yohannlai/RAG_picsou.git
cd RAG_picsou

# Création de l'environnement virtuel
python3 -m venv venv

# Activation de l'environnement (MacOS/Linux)
source venv/bin/activate

# Activation de l'environnement (Windows)
.\venv\Scripts\activate
~~~

### 3. Installer les dépendances
Ne gaspillez pas d'électricité, installez toutes les bibliothèques requises pour le mode local d'un coup :
~~~bash
pip install -r requirements.txt
~~~

### 4. Configuration des clés API
Créez un fichier nommé `.env` à la racine du projet (il sera ignoré par Git grâce au `.gitignore`) et ajoutez-y vos clés API OpenRouter et Hugging Face :
~~~text
OPENROUTER_API_KEY=votre_cle_api_openrouter_ici
HUGGINGFACEHUB_API_TOKEN=votre_cle_api_huggingface_ici
~~~

---

## 💬 Utilisation

Vous avez deux façons d'interroger Balthazar Picsou :

### Option A : L'Interface Web (Recommandée)
Pour profiter de l'interface graphique responsive avec les suggestions de questions intégrées :
~~~bash
python3 main_flask.py
~~~
Ouvrez ensuite votre navigateur à l'adresse suivante : **`http://127.0.0.1:5000`**

### Option B : Le Terminal (Pour les puristes)
Exécutez le script principal. Le système va charger la base vectorielle et ouvrir l'invite de commande interactive de Picsou :
~~~bash
python3 main.py
~~~
> **🦆 Ta question, misérable fouineur :** Qui est Gontran Bonheur ?

Le RAG cherchera les morceaux les plus pertinents dans les fichiers `.txt` et générera une réponse contextuelle.
*Note : Tapez `quit`, `exit` ou `q` pour fermer le coffre et quitter le programme proprement.*

---
*Un sou est un sou !*