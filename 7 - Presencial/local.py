import requests 
import json 
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords

import time

# Baixar stopwords do NLTK 
nltk.download("punkt") 
nltk.download("stopwords") 
nltk.download("punkt_tab")

# URL do Ollama rodando localmente (após ser instalado, esta URL é criada) 
OLLAMA_URL = "http://localhost:11434/api/generate"

# Dicionário de palavras-chave e fluxos de trabalho 
fluxos = {
    "correr": "Atividades e Planos de Corrida",
    "pedalar": "Atividades e Planos de Pedal",
    "nadar": "Atividades e Planos de Natação", 
}

def extrair_palavras_chave(texto): 
    """Extrai palavras-chave de uma pergunta removendo stopwords.""" 
    tokens = word_tokenize(texto.lower())  # Tokenização e conversão para minúsculas 
    stop_words = set(stopwords.words("portuguese")) 
    palavras_chave = [t for t in tokens if t.isalnum() and t not in stop_words] 
    return palavras_chave

def obter_resposta_llama(pergunta): 
    """Envia a pergunta para o LLaMA e retorna a resposta.""" 
    payload = { 
        "model": "qwen3:0.6b",
#        "model": "qwen3:4b",
        "prompt": pergunta, 
        "stream": False 
    }
    resposta = requests.post(OLLAMA_URL, json=payload) 
    return resposta.json()["response"] if resposta.status_code == 200 else "Erro na resposta"

def determinar_fluxo(pergunta): 
    """Determina o fluxo de trabalho com base nas palavras-chave da pergunta.""" 
    palavras_chave = extrair_palavras_chave(pergunta) 
    for palavra in palavras_chave: 
        if palavra in fluxos: 
            return f"Fluxo identificado: {fluxos[palavra]}" 
    return "Nenhum fluxo específico identificado."

if __name__ == "__main__": 
    while True: 
        pergunta = input("Digite sua pergunta (ou 'sair' para encerrar): ") 
        if pergunta.lower() == "sair": 
            break 
        
        inicio = time.time()
        fluxo = determinar_fluxo(pergunta) 
        resposta = obter_resposta_llama(pergunta) 
         
        print(fluxo) 
        print("Resposta do Assistente:", resposta)
        fim = time.time()

        print(f"Tempo de execução: {fim - inicio} segundos")