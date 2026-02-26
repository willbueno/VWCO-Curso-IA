from fastapi import FastAPI 
from pydantic import BaseModel 
import requests 
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    UnstructuredExcelLoader,
    CSVLoader
)
from langchain_chroma.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from typing import List

from fastapi.middleware.cors import CORSMiddleware
 
nltk.download("punkt") 
nltk.download("stopwords") 
 
# Inicializa a aplicação FastAPI 
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=("*"),
    allow_credentials=True,
    allow_methods=("*"),
    allow_headers=("*"),
)

# Dicionário de fluxos de trabalho com múltiplas palavras-chave 
fluxos = { 
    "reunião": ["reunião", "agenda", "marcar encontro", "reuniões"], 
    "relatório": ["relatório", "gerar relatório", "criar relatório", "relatórios"], 
    "projeto": ["projeto", "gestão de projetos", "gerenciamento de projetos"], 
    "financeiro": ["financeiro", "contas", "relatório financeiro", "finanças"], 
}

OLLAMA_URL = "http://localhost:11434/api/generate"  # Substitua com a URL do Ollama 
 
# Pasta onde estão os documentos 
PASTA_DOCUMENTOS = "documents" 
 
# Carregar e indexar documentos da pasta 
def carregar_documentos(): 
    documentos = [] 
    for arquivo in os.listdir(PASTA_DOCUMENTOS): 
        caminho_arquivo = os.path.join(PASTA_DOCUMENTOS, arquivo) 
        if arquivo.endswith(".pdf"): 
            loader = PyPDFLoader(caminho_arquivo) 
        elif arquivo.endswith(".txt"): 
            loader = TextLoader(caminho_arquivo) 
        else: 
            continue 
        documentos.extend(loader.load())
    return documentos 
 
# Criar o índice vetorial para consulta eficiente 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
index = Chroma.from_documents(carregar_documentos(), embeddings) 
 
class Pergunta(BaseModel): 
    pergunta: str

def extrair_palavras_chave(texto: str) -> List[str]: 
    """Extrai palavras-chave da pergunta""" 
    stop_words = set(stopwords.words("portuguese")) 
    tokens = word_tokenize(texto.lower())  # Tokeniza e converte em minúsculas 
    palavras_chave = [t for t in tokens if t.isalnum() and t not in stop_words] 
    return palavras_chave 
 
def determinar_fluxo(palavras_chave: List[str]) -> str: 
    """Identifica o fluxo de trabalho baseado nas palavras-chave""" 
    for fluxo, palavras in fluxos.items(): 
        # Verifica se alguma palavra-chave do fluxo aparece nas palavras extraídas 
        if any(palavra in palavras_chave for palavra in palavras): 
            return f"Fluxo relacionado a: {fluxo.capitalize()}" 
    return "Nenhum fluxo específico identificado."

# Função para obter resposta do LLaMA com a consulta dos documentos 
def obter_resposta_llama(pergunta: str) -> str: 
    # Primeiro, buscar nos documentos usando o índice 
    resultado = index.similarity_search(pergunta, k=3)
         
    # O contexto da busca será usado para gerar a resposta final 
    contexto = "\n\n".join([doc.page_content for doc in resultado]) 
     
    # Usar LLaMA para gerar a resposta final 
    payload = { 
#        "model": "llama3.2",
        "model": "qwen3:0.6b",
        "prompt": f"{contexto}\n\nPergunta: {pergunta}", 
        "stream": False 
    } 
    resposta = requests.post(OLLAMA_URL, json=payload) 
    return resposta.json().get("response", "Erro ao obter resposta") 
 
@app.post("/pergunta/") 
async def fazer_pergunta(pergunta: Pergunta): 
    """Processa a pergunta, envia ao LLaMA e retorna a resposta""" 
    palavras_chave = extrair_palavras_chave(pergunta.pergunta) 
    fluxo = determinar_fluxo(palavras_chave) 
     
    # Obter resposta do LLaMA com consulta aos documentos 
    resposta_llama = obter_resposta_llama(pergunta.pergunta) 
     
    return {"fluxo": fluxo, "resposta": resposta_llama} 