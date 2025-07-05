import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor
import re
import numpy as np
import os

def fetch_text_from_url(url):
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Sadece ana içerik metnini çek
        texts = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li']):
            txt = tag.get_text(strip=True)
            if txt:
                texts.append(txt)
        return '\n'.join(texts)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def chunk_text(text, max_tokens=120):
    # Daha büyük chunk'lar oluştur
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    chunk = ''
    for sent in sentences:
        if len((chunk + sent).split()) > max_tokens:
            chunks.append(chunk.strip())
            chunk = sent
        else:
            chunk += ' ' + sent
    if chunk:
        chunks.append(chunk.strip())
    return [c for c in chunks if len(c.split()) > 8]  # Minimum kelime sayısını artır

def setup_database():
    database_url = os.getenv('DATABASE_URL', 'postgresql://localhost/boundless')
    conn = psycopg2.connect(database_url)
    cursor = conn.cursor()
    
    # pgvector extension'ı ekle
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Tablo oluştur
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS boundless_chunks (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            url TEXT NOT NULL,
            embedding vector(384),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Eski verileri temizle
    cursor.execute("DELETE FROM boundless_chunks;")
    
    conn.commit()
    cursor.close()
    conn.close()
    print("Database setup completed")

def main():
    urls = [
        # Ana sayfa ve temel dokümantasyon
        'https://beboundless.xyz/',
        'https://docs.beboundless.xyz/developers/what',
        'https://docs.beboundless.xyz/developers/proof-lifecycle',
        'https://docs.beboundless.xyz/developers/why',
        'https://docs.beboundless.xyz/developers/core-concepts',
        
        # Ecosystem - 25+ protokol entegrasyonu
        'https://beboundless.xyz/ecosystem',
        
        # Blog - güncel gelişmeler ve teknik yazılar
        'https://beboundless.xyz/blog',
        
        # Prover dokümantasyonu - node çalıştırma
        'https://docs.beboundless.xyz/provers/quick-start',
        'https://docs.beboundless.xyz/provers/who-should-run',
        'https://docs.beboundless.xyz/provers/requirements',
        
        # Ek teknik dokümanlar
        'https://docs.beboundless.xyz/developers/quick-start',
        'https://docs.beboundless.xyz/developers/build-a-program',
        'https://docs.beboundless.xyz/developers/request-a-proof',
        'https://docs.beboundless.xyz/developers/use-a-proof',
        
        # Whitepaper ve teknik dökümanlar
        'https://read.beboundless.xyz/',
    ]
    
    # Database setup
    setup_database()
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    database_url = os.getenv('DATABASE_URL', 'postgresql://localhost/boundless')
    conn = psycopg2.connect(database_url)
    cursor = conn.cursor()
    
    total_chunks = 0
    
    for url in urls:
        print(f"[+] {url} çekiliyor...")
        text = fetch_text_from_url(url)
        chunks = chunk_text(text)
        print(f"[+] {len(chunks)} adet chunk bulundu.")
        
        if chunks:
            embeddings = embedder.encode(chunks)
            embeddings = np.array(embeddings, dtype=np.float32)
            
            for i, chunk in enumerate(chunks):
                cursor.execute("""
                    INSERT INTO boundless_chunks (content, url, embedding)
                    VALUES (%s, %s, %s)
                """, (chunk, url, embeddings[i].tolist()))
                total_chunks += 1
            
            print(f"[+] {url} için embedding ve kayıt tamamlandı.")
        else:
            print(f"[!] {url} için chunk bulunamadı.")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"[+] Toplam {total_chunks} chunk PostgreSQL'e kaydedildi.")

if __name__ == "__main__":
    main() 