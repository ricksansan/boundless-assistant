import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
import re
import numpy as np

def fetch_text_from_url(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    # Sadece ana içerik metnini çek (örnek: tüm <p> ve <h*> tag'leri)
    texts = []
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li']):
        txt = tag.get_text(strip=True)
        if txt:
            texts.append(txt)
    return '\n'.join(texts)

def chunk_text(text, max_tokens=40):
    # Basit cümle bazlı chunking (daha küçük parçalara bölecek)
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
    return [c for c in chunks if len(c.split()) > 5]  # çok kısa chunk'ları atla

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
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.HttpClient(host="chroma", port=8000)
    collection = chroma_client.get_or_create_collection('boundless_docs')
    all_ids = collection.get()['ids']
    if all_ids:
        collection.delete(ids=all_ids)
    for url in urls:
        print(f"[+] {url} çekiliyor...")
        text = fetch_text_from_url(url)
        chunks = chunk_text(text)
        print(f"[+] {len(chunks)} adet chunk bulundu.")
        embeddings = embedder.encode(chunks)
        embeddings = np.array(embeddings, dtype=np.float32)
        for i, chunk in enumerate(chunks):
            meta = {"url": url, "chunk_id": i}
            collection.add(
                documents=[chunk],
                embeddings=[embeddings[i]],
                metadatas=[meta],
                ids=[f"{url}_chunk_{i}"]
            )
        print(f"[+] {url} için embedding ve kayıt tamamlandı.")

if __name__ == "__main__":
    main() 