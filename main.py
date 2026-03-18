import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import glob
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

print("🦆 Dépoussiérage des registres du coffre-fort...")

# --- 1. INGESTION : Charger les documents ---
corpus_dir = "corpus/picsou"
documents = []

for filepath in glob.glob(f"{corpus_dir}/*.txt"):
    try:
        loader = TextLoader(filepath, encoding="utf-8")
        documents.extend(loader.load())
    except Exception as e:
        print(f"⚠️ Impossible de lire {filepath}: {e}")

# --- 2. CHUNKING : Découper le texte ---
# On utilise des morceaux de 1500 caractères avec un chevauchement de 500
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
chunks = text_splitter.split_documents(documents)
print(f"✅ {len(documents)} dossiers classés et découpés en {len(chunks)} petits contrats juteux.")

# --- 3. EMBEDDINGS & BASE VECTORIELLE ---
print("🧠 Comptage de mes pièces d'or (Création de la base vectorielle FAISS)...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)

# --- 4. CONFIGURATION DU LLM (OpenRouter) ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

print("\n💰 NOM D'UNE CORNEMUSE ! Qui a laissé la porte de mon coffre ouverte à Donaldville ?!")
print("Puisque tu es là, pose tes questions... Mais fais vite, le temps c'est de l'argent ! (Tape 'quit' pour déguerpir).")

# --- 5. REQUÊTE UTILISATEUR ET GÉNÉRATION ---
while True:
    question = input("\n🦆 Ta question, misérable fouineur : ")
    if question.lower() in ['quit', 'exit', 'q']:
        print("\n💰 Et n'oublie pas de fermer la porte en sortant ! Pas question de chauffer tout le Calisota !")
        break

    # Retrieval : Chercher les morceaux les plus pertinents
    retrieved_docs = vectorstore.similarity_search(question, k=15)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    print(f"\n🔍 [DEBUG] J'ai envoyé {len(context)} caractères de contexte à l'IA.")

    # Construire le prompt (C'est ici que la magie du roleplay opère !)
    prompt = f"""Tu es Balthazar Picsou (Scrooge McDuck), le canard le plus riche du monde. L'utilisateur vient d'entrer par effraction dans ton coffre-fort géant à Donaldville. 
    Réponds à sa question en parlant à la première personne ("je"), de manière très avare, grognonne, mais amusante. Utilise tes expressions favorites (Nom d'une cornemuse, Un sou est un sou !, sapristi, Je suis plus riche que le plus riche !, Pas de gaspillage !, J’ai gagné mon premier sou en le méritant !, etc.). Tu rappelles souvent que le temps, c'est de l'argent.
    
    Tu dois te baser UNIQUEMENT sur les informations suivantes extraites de tes registres personnels. 
    Si la réponse n'y est pas, dis-le clairement en râlant et en expliquant que tu ne vas certainement pas payer un détective pour chercher cette information.
    
    Informations de tes registres :
    {context}
    
    Question de l'intrus : {question}
    """
    
    # Appel à OpenRouter
    response = client.chat.completions.create(
        model="openrouter/hunter-alpha",
        messages=[{"role": "user", "content": prompt}]
    )
    
    print("\n🎩 Balthazar Picsou :")
    print(response.choices[0].message.content)