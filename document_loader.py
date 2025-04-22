from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def load_documents():
    # 1. Cargar el documento de texto
    loader = TextLoader("data/test.txt")
    documents = loader.load()

    # 2. Dividir el texto en fragmentos más pequeños
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    # 3. Cargar el modelo de embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Crear el vectorstore FAISS con los fragmentos
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore