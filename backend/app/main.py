from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
import re
import json

app = FastAPI()

# Embedding modeli (local, ücretsiz)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory storage
boundless_data = []

# Hazır soru-cevap listesi
PREDEFINED_QA = {
    "What is Boundless?": {
        "answer": "Boundless is a decentralized protocol for generating and verifying zero-knowledge (ZK) proofs. It enables anyone to request, produce, and use ZK proofs in a trustless marketplace, powered by RISC Zero zkVM.",
        "source_url": "https://docs.beboundless.xyz/developers/what",
        "source": "predefined"
    },
    "What is the proof lifecycle?": {
        "answer": "The proof lifecycle has 6 steps: 1) Program Development, 2) Request Submission, 3) Prover Bidding, 4) Proof Generation, 5) Proof Settlement, 6) Proof Utilization. Each step is decentralized and verifiable.",
        "source_url": "https://docs.beboundless.xyz/developers/proof-lifecycle",
        "source": "predefined"
    },
    "How to run a prover node?": {
        "answer": "To run a prover node: 1) Use a powerful machine (32GB+ RAM, modern GPU), 2) Install RISC Zero, 3) Set up Boundless CLI tools, 4) Listen for proof requests and submit results. See the official quick start guide for details.",
        "source_url": "https://docs.beboundless.xyz/provers/quick-start",
        "source": "predefined"
    },
    "What is ZK mining?": {
        "answer": "ZK mining is the process of generating zero-knowledge proofs using computational resources. Provers compete to solve proof requests and earn rewards for successful submissions.",
        "source_url": "https://docs.beboundless.xyz/provers/quick-start",
        "source": "predefined"
    },
    "What is the Boundless SDK?": {
        "answer": "The Boundless SDK allows developers to interact with the protocol, submit proof requests, and use proofs in their applications. SDKs are available for JavaScript, Python, and Rust.",
        "source_url": "https://docs.beboundless.xyz/developers/quick-start",
        "source": "predefined"
    },
    "What projects are in the ecosystem?": {
        "answer": "The Boundless ecosystem includes 25+ projects: Hibachi Exchange (DEX), Lido (Staking), Uniswap, Aave, Compound, MakerDAO, Curve, Balancer, and more. All leverage ZK proofs for enhanced security.",
        "source_url": "https://beboundless.xyz/ecosystem",
        "source": "predefined"
    },
    "What are the security features?": {
        "answer": "Boundless security: 1) ZK proofs for mathematical security, 2) Decentralized architecture, 3) Open-source code and audits, 4) Multisig management, 5) Bug bounty program, 6) Continuous updates.",
        "source_url": "https://docs.beboundless.xyz/developers/core-concepts",
        "source": "predefined"
    },
    "What is the tokenomics?": {
        "answer": "The ticker is $ZKC",
        "source_url": "https://beboundless.xyz/",
        "source": "predefined"
    }
}

def search_web(query: str) -> str:
    """DuckDuckGo ile web araması yap"""
    try:
        # DuckDuckGo Instant Answer API
        search_url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
        response = requests.get(search_url, timeout=10)
        data = response.json()
        
        if data.get('Abstract'):
            return data['Abstract']
        elif data.get('Answer'):
            return data['Answer']
        elif data.get('RelatedTopics') and len(data['RelatedTopics']) > 0:
            return data['RelatedTopics'][0].get('Text', 'No information found.')
        else:
            return "I couldn't find specific information about that on the web."
            
    except Exception as e:
        print(f"Web search error: {e}")
        return "Sorry, I couldn't search the web at the moment."

def fetch_and_process_data():
    """Boundless verilerini çek ve işle - Gelişmiş scraping"""
    urls = [
        'https://beboundless.xyz/',
        'https://docs.beboundless.xyz/developers/what',
        'https://docs.beboundless.xyz/developers/proof-lifecycle',
        'https://docs.beboundless.xyz/developers/why',
        'https://docs.beboundless.xyz/developers/core-concepts',
        'https://docs.beboundless.xyz/provers/quick-start',
        'https://docs.beboundless.xyz/provers/who-should-run',
        'https://docs.beboundless.xyz/provers/requirements',
        'https://docs.beboundless.xyz/developers/quick-start',
        'https://docs.beboundless.xyz/developers/build-a-program',
        'https://docs.beboundless.xyz/developers/request-a-proof',
        'https://docs.beboundless.xyz/developers/use-a-proof',
        'https://beboundless.xyz/ecosystem',
        'https://beboundless.xyz/blog',
        'https://read.beboundless.xyz/',
    ]
    
    for url in urls:
        try:
            print(f"Fetching: {url}")
            resp = requests.get(url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Ana içerik metnini çek - daha kapsamlı
            texts = []
            
            # Başlıkları al
            for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5']):
                txt = tag.get_text(strip=True)
                if len(txt) > 5 and len(txt) < 200:
                    texts.append(f"Title: {txt}")
            
            # Paragrafları al
            for tag in soup.find_all(['p', 'li', 'div']):
                txt = tag.get_text(strip=True)
                if len(txt) > 20 and len(txt) < 1000:  # Daha uzun metinler
                    texts.append(txt)
            
            # Chunk'lara böl - daha küçük chunk'lar
            full_text = ' '.join(texts)
            sentences = re.split(r'(?<=[.!?]) +', full_text)
            
            chunk = ''
            for sent in sentences:
                if len((chunk + sent).split()) > 30:  # 30 kelimelik chunk'lar
                    if chunk.strip():
                        boundless_data.append({
                            'content': chunk.strip(),
                            'url': url
                        })
                    chunk = sent
                else:
                    chunk += ' ' + sent
            
            if chunk.strip():
                boundless_data.append({
                    'content': chunk.strip(),
                    'url': url
                })
                
        except Exception as e:
            print(f"Error fetching {url}: {e}")

# Uygulama başladığında verileri yükle
@app.on_event("startup")
async def startup_event():
    print("Loading Boundless data...")
    fetch_and_process_data()
    print(f"Loaded {len(boundless_data)} chunks")

@app.get("/")
def root():
    return {"message": "Boundless Assistant Backend Çalışıyor!", "chunks_loaded": len(boundless_data)}

@app.get("/predefined-questions")
def get_predefined_questions():
    """Hazır soruları döndür"""
    return {"questions": list(PREDEFINED_QA.keys())}

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")
    
    if not question:
        return {"answer": "Please ask a question.", "source_url": None}
    
    try:
        # Soruyu normalize et
        normalized_question = question.lower().strip()
        question_words = set(re.findall(r'\w+', normalized_question))
        
        # 1. Önce hazır soru-cevap listesinde akıllı ara
        best_predefined_match = None
        best_predefined_score = 0
        
        for predefined_q, predefined_a in PREDEFINED_QA.items():
            predefined_words = set(re.findall(r'\w+', predefined_q.lower()))
            
            # Anahtar kelime eşleşmesi
            keyword_overlap = len(question_words & predefined_words)
            
            # Tam eşleşme kontrolü
            if normalized_question in predefined_q.lower() or predefined_q.lower() in normalized_question:
                keyword_overlap += 5  # Bonus puan
            
            # Özel kelime eşleşmeleri
            special_keywords = {
                'boundless': 3,
                'proof': 2,
                'lifecycle': 2,
                'prover': 2,
                'node': 2,
                'zk': 2,
                'mining': 2,
                'sdk': 2,
                'ecosystem': 2,
                'security': 2,
                'tokenomics': 3,
                'token': 2,
                'zkc': 3
            }
            
            for word, weight in special_keywords.items():
                if word in normalized_question and word in predefined_q.lower():
                    keyword_overlap += weight
            
            if keyword_overlap > best_predefined_score:
                best_predefined_score = keyword_overlap
                best_predefined_match = predefined_a
        
        # Eğer yeterince iyi bir predefined match varsa, onu döndür
        if best_predefined_score >= 2:
            return {
                "answer": best_predefined_match["answer"],
                "source_url": best_predefined_match["source_url"],
                "similarity_score": float(best_predefined_score),
                "source": "predefined"
            }
        
        # 2. Dokümanda ara - daha akıllı arama
        if boundless_data:
            question_embedding = embedder.encode([question])[0]
            best_match = None
            best_score = -1
            
            for chunk in boundless_data:
                # Chunk filtreleme - çok uzun veya çok kısa chunk'ları atla
                if len(chunk['content']) < 20 or len(chunk['content']) > 800:
                    continue
                
                # Title chunk'larını atla
                if chunk['content'].startswith('Title:'):
                    continue
                
                chunk_embedding = embedder.encode([chunk['content']])[0]
                similarity = np.dot(question_embedding, chunk_embedding) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(chunk_embedding)
                )
                
                # Anahtar kelime eşleşmesi
                chunk_words = set(re.findall(r'\w+', chunk['content'].lower()))
                keyword_overlap = len(question_words & chunk_words)
                
                # Özel kelime bonusları
                special_bonus = 0
                for word, weight in special_keywords.items():
                    if word in normalized_question and word in chunk['content'].lower():
                        special_bonus += weight
                
                # Toplam skor: embedding + keyword + bonus
                score = similarity + (keyword_overlap * 0.3) + (special_bonus * 0.2)
                
                if score > best_score:
                    best_score = score
                    best_match = chunk
            
            if best_match and best_score > 0.3:  # Minimum similarity threshold
                # Cevabı daha akıllıca formatla
                answer = best_match['content'].strip()
                
                # Çok uzunsa kısalt
                if len(answer) > 300:
                    sentences = answer.split('.')
                    shortened = ''
                    for sentence in sentences:
                        if len(shortened + sentence) < 300:
                            shortened += sentence + '.'
                        else:
                            break
                    answer = shortened.strip()
                
                return {
                    "answer": answer,
                    "source_url": best_match['url'],
                    "similarity_score": float(best_score),
                    "source": "documentation"
                }
        
        # 3. Son çare: web araması
        web_answer = search_web(question + " Boundless blockchain protocol")
        
        return {
            "answer": web_answer,
            "source_url": "https://duckduckgo.com",
            "similarity_score": 0.0,
            "source": "web_search"
        }
            
    except Exception as e:
        print(f"Error: {e}")
        return {"answer": "Sorry, an error occurred.", "source_url": None}

@app.post("/health")
async def health_check():
    return {"status": "healthy", "chunks_loaded": len(boundless_data)} 