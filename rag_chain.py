from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from document_loader import load_documents

def create_rag_chain():
    # 1. Cargar documentos y FAISS
    vectorstore = load_documents()
    retriever = vectorstore.as_retriever()

    # 2. Cargar el modelo de lenguaje (modelo pequeño)
    model_name = "tiiuae/falcon-rw-1b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

    # 3. Integrar modelo + documentos usando LangChain
    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain