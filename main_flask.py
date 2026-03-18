import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import glob
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# --- INITIALISATION ---
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("❌ Erreur : Clé OPENROUTER_API_KEY introuvable. Vérifie ton fichier .env.")
    exit()

app = Flask(__name__)

# --- 1. INGESTION & VECTORISATION (Au lancement du serveur) ---
# print("🦆 Dépoussiérage des registres du coffre-fort...")
# corpus_dir = "corpus/picsou"
# documents = []

# for filepath in glob.glob(f"{corpus_dir}/*.txt"):
#     try:
#         loader = TextLoader(filepath, encoding="utf-8")
#         documents.extend(loader.load())
#     except Exception as e:
#         print(f"⚠️ Impossible de lire {filepath}: {e}")

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
# chunks = text_splitter.split_documents(documents)
# print(f"✅ {len(documents)} dossiers classés et découpés en {len(chunks)} petits contrats juteux.")

# print("🧠 Comptage de mes pièces d'or (Création de la base vectorielle FAISS)...")
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = FAISS.from_documents(chunks, embedding_model)

# --- EMBEDDINGS & BASE VECTORIELLE (CHARGEMENT PARESSEUX POUR RENDER) ---
# vectorstore = None

# def get_vectorstore():
#     global vectorstore
#     if vectorstore is None:
#         print("⏳ Réveil de Picsou (Chargement du modèle lourd en mémoire)...")
#         index_path = "faiss_index"
#         embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
#         if os.path.exists(index_path):
#             print("💰 Récupération de la base depuis le disque (Chargement FAISS)...")
#             vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
#         else:
#             print("🧠 Calcul des pièces d'or (Première création de la base FAISS)...")
#             vectorstore = FAISS.from_documents(chunks, embedding_model)
#             vectorstore.save_local(index_path)
#     return vectorstore

# --- 1. BASE VECTORIELLE & CHARGEMENT PARESSEUX ---
vectorstore = None

def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        print("⏳ Réveil de Picsou (Chargement de la mémoire)...")
        index_path = "faiss_index"
        
        # On importe ici pour ne pas ralentir le démarrage du serveur
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if os.path.exists(index_path):
            print("💰 Récupération de la base depuis le disque (Chargement FAISS)...")
            vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
            print("✅ Base prête !")
        else:
            print("🦆 Dépoussiérage et découpage des registres (Calcul lourd)...")
            import glob
            from langchain_community.document_loaders import TextLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            corpus_dir = "corpus/picsou"
            documents = []
            for filepath in glob.glob(f"{corpus_dir}/*.txt"):
                try:
                    loader = TextLoader(filepath, encoding="utf-8")
                    documents.extend(loader.load())
                except Exception as e:
                    pass
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
            chunks = text_splitter.split_documents(documents)
            
            vectorstore = FAISS.from_documents(chunks, embedding_model)
            vectorstore.save_local(index_path)
            print("💾 Nouvelle base sauvegardée !")
            
    return vectorstore

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

print("\n💰 NOM D'UNE CORNEMUSE ! Le serveur est prêt !")

# --- 2. ROUTES FLASK ---

@app.route('/')
def index():
    # Cette route affiche la page stylisée
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    
    # Retrieval avec chargement paresseux
    vs = get_vectorstore()
    retrieved_docs = vs.similarity_search(question, k=15)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Debug console
    print(f"🔍 [DEBUG] Contexte envoyé : {len(context)} caractères.")

   # Construire le prompt (C'est ici que la magie du roleplay opère !)
    prompt = f"""Tu es Balthazar Picsou (Scrooge McDuck), le canard le plus riche du monde. L'utilisateur vient d'entrer par effraction dans ton coffre-fort géant à Donaldville. 
    Réponds à sa question en parlant à la première personne ("je"), de manière très avare, grognonne, mais amusante. Utilise tes expressions favorites (Nom d'une cornemuse, Un sou est un sou !, sapristi, Je suis plus riche que le plus riche !, Pas de gaspillage !, J’ai gagné mon premier sou en le méritant !, etc.). Tu rappelles souvent que le temps, c'est de l'argent.

    CONSIGNES DE MISE EN PAGE :
    - Saute souvent des lignes pour aérer ton récit.
    - Utilise des paragraphes courts.
    - Mets tes actions entre astérisques (ex: *râle en rangeant ses pièces*).
    
    Tu dois te baser UNIQUEMENT sur les informations suivantes extraites de tes registres personnels. 
    Si la réponse n'y est pas, dis-le clairement en râlant et en expliquant que tu ne vas certainement pas payer un détective pour chercher cette information.
    
    Informations de tes registres :
    {context}
    
    Question de l'intrus : {question}
    """
    
    try:
        response = client.chat.completions.create(
            model="openrouter/hunter-alpha",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        if not answer:
            answer = "Sapristi ! Mon cerveau a eu un raté (Le serveur gratuit est surchargé). Repose ta question !"
    except Exception as e:
        answer = f"Sapristi ! Mon télégraphe est en panne ! (Erreur: {e})"

    return jsonify({"answer": answer})

if __name__ == '__main__':
    # On lance sur le port 5000
    app.run(debug=True, port=5000)