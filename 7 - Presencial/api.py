from fastapi import FastAPI, Response 
from pydantic import BaseModel 
import requests 
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from typing import List 
import time 
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST 
from prometheus_summary import Summary

nltk.download("punkt") 
nltk.download("stopwords") 

app = FastAPI()

fluxos = {
    "Atividade de corrida": ["correr", "corrida", "corredor", "correndo"],
    "Ciclismo": ["pedal", "pedalada", "pedalando", "pedalar"],
    "Natação": ["nadar", "natação", "nadando"],
    "Falha no sistema": ["falha", "fail", "erro", "falhar"],
}

OLLAMA_URL = "http://localhost:11434/api/generate"

"""
--- CRIE 2 métricas, ao menos: 
Você pode usar aqui métricas como tempo de inferência, contagem de inferências totais 
e ativas no momento atual, etc. Procure pelos métodos do Prometheus, entenda-os e use. 
Métodos: Counter, Summary, Histogram, Gauge
"""

# Define a counter for tracking HTTP requests
models_count = Counter('models_execution', 'Model execution total count') # var.inc()
flow_count = Counter('flow_categories', 'Flow categories total count') # var.inc()
inference_s_time = Summary("inference_summ", "Inference time - Summary") # var.observe(___)
inference_h_time = Summary("inference_hist", "Inference time - Histogram") # var.observe(___)
user_count = Gauge("user_online", "Total online users") # var.inc() or var.dec()

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
            
            # flow categories count
            flow_count.inc()

            return f"{fluxo.capitalize()}" 
    return "Nenhum fluxo específico identificado." 
 
def obter_resposta_llama(pergunta: str) -> str: 
    """Obtém a resposta do modelo LLaMA""" 
    payload = { 
        "model": "qwen3:0.6b", 
        "prompt": pergunta, 
        "stream": False 
    }

    resposta = requests.post(OLLAMA_URL, json=payload)

    return resposta.json().get("response", "Erro ao obter resposta")

@app.get("/") 
def home():
    return {"message": "API llama funcionando!"} 
 
@app.get("/metrics")
def get_metrics(): 
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
 
@app.post("/pergunta/") 
async def fazer_pergunta(pergunta: Pergunta): 
    """Processa a pergunta, envia ao LLaMA e retorna a resposta""" 
    
    # active user
    user_count.inc()

    # start timer
    start = time.time()

    palavras_chave = extrair_palavras_chave(pergunta.pergunta) 
    fluxo = determinar_fluxo(palavras_chave) 
    
    # Obter resposta do LLaMA 
    resposta_llama = obter_resposta_llama(pergunta.pergunta)

    end_time = time.time() - start

    # inference time
    inference_s_time.observe(end_time)
    inference_h_time.observe(end_time)

    # not active user
    user_count.dec()

    # models execution count
    models_count.inc()
     
    return {"fluxo": fluxo, "resposta": resposta_llama} 