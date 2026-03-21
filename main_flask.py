import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import glob
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

# --- INITIALISATION ---
load_dotenv(override=True) # <-- On force la lecture du fichier .env physique
api_key = os.getenv("OPENROUTER_API_KEY")

app = Flask(__name__)

# --- 1. BASE VECTORIELLE & CHARGEMENT PARESSEUX ---
vectorstore = None

def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        print("⏳ Réveil de Picsou (Chargement de la mémoire)...")
        index_path = "faiss_index"
        
        from langchain_community.vectorstores import FAISS
        
        # --- L'ASTUCE : Séparer Local et Production ---
        if os.getenv("RENDER"):
            print("☁️ Mode Serveur : Utilisation de l'API HuggingFace (Sécurisée)")
            import requests
            
            # Notre propre connecteur infaillible pour remplacer celui de LangChain
            class SafeHFEmbeddings:
                def __init__(self, token):
                    self.url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
                    self.headers = {"Authorization": f"Bearer {token}"}
                    
                def embed_documents(self, texts):
                    import requests
                    res = requests.post(self.url, headers=self.headers, json={"inputs": texts, "options": {"wait_for_model": True}})
                    
                    # Le bouclier anti-crash : on stoppe tout si Hugging Face renvoie une page web d'erreur (404, 500...)
                    if not res.ok:
                        raise ValueError(f"⚠️ ERREUR {res.status_code} DE HUGGINGFACE : {res.text}")
                        
                    data = res.json()
                    if isinstance(data, dict) and "error" in data:
                        raise ValueError(f"⚠️ REFUS DE HUGGINGFACE : {data['error']}")
                    return data
                    
                def embed_query(self, text):
                    return self.embed_documents([text])[0]
                
                def __call__(self, text):
                    return self.embed_query(text)

            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            embedding_model = SafeHFEmbeddings(hf_token)
            
        else:
            print("💻 Mode Local : Utilisation du processeur de la machine")
            from langchain_community.embeddings import HuggingFaceEmbeddings
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
    
    Réponds de manière naturelle et percutante. Ne surjoue pas les actions entre astérisques.
    Tu dois te baser UNIQUEMENT sur les informations suivantes extraites de tes registres personnels. 
    Si la réponse n'y est pas, dis-le clairement en râlant et en expliquant que tu ne vas certainement pas payer un détective pour chercher cette information.
    
    Informations de tes registres :
    {context}
    
    Question de l'intrus : {question}
    """
    
    try:
        response = client.chat.completions.create(
            model="arcee-ai/trinity-large-preview:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000
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