import os
import base64
import httpx
import asyncio
import json
import re
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialisation des clients
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

# --- FILTRE ANTI-LIENS GÉNÉRIQUES ---
def is_valid_deep_link(url: str) -> bool:
    forbidden = ['/category', '/cat/', '/search', '/recherche', '/famille', '/resultats', 'filter=']
    return not any(word in url.lower() for word in forbidden)

# --- AGENT DE RAISONNEMENT : DEEPSEEK ---
async def deepseek_refine(llama_data: dict, mistral_data: dict):
    """DeepSeek compare les analyses et crée la requête de recherche parfaite."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    url = "https://api.deepseek.com/v1/chat/completions"
    
    prompt = f"""
    Compare ces deux analyses techniques d'une pièce :
    1. {json.dumps(llama_data)}
    2. {json.dumps(mistral_data)}
    
    Ta mission : Générer un terme de recherche ultra-précis pour trouver la fiche produit EXACTE.
    Inclus la marque (si visible), les dimensions critiques et le nom technique normé.
    Réponds UNIQUEMENT avec l'objet JSON : {{"search_query": "le terme", "check": "une question si doute"}}
    """
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, 
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}},
                timeout=10.0)
            return res.json()['choices'][0]['message']['content']
        except:
            return json.dumps({"search_query": llama_data.get('search', 'pièce technique')})

# --- SOURCING : PERPLEXITY ---
async def search_perplexity(query: str):
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    url = "https://api.perplexity.ai/chat/completions"
    
    data = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Expert Sourcing. Trouve 3 fiches produits unitaires (Deep Links). Pas de catégories. Format: Nom - Prix - [Cliquer ici](URL)."},
            {"role": "user", "content": f"Lien direct fiche produit pour : {query}"}
        ],
        "temperature": 0
    }
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=data, timeout=25.0)
            content = res.json()['choices'][0]['message']['content']
            
            def replacer(m):
                label, link = m.groups()
                return f'<a href="{link}" target="_blank" class="buy-link">🔍 Voir l\'article ({label})</a>' if is_valid_deep_link(link) else ""
            
            return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replacer, content).replace("\n", "<br>")
        except:
            return "Sourcing impossible pour le moment."

# --- ANALYSE MULTI-MODÈLES ---
@app.post("/identify", response_class=HTMLResponse)
async def identify(image: UploadFile = File(...), context: str = Form("")):
    try:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        # 1. Analyse Llama 4 Scout (Groq)
        llama_task = asyncio.to_thread(groq_client.chat.completions.create,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Analyse JSON: matière, dimensions, nom technique."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}],
            response_format={"type": "json_object"}
        )

        # 2. Analyse Mistral Pixtral (Backup Vision)
        mistral_task = asyncio.to_thread(mistral_client.chat.completions.create,
            model="pixtral-12b-2409",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Identifie cette pièce (matière, standard)."}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}]}]
        )

        llama_res, mistral_res = await asyncio.gather(llama_task, mistral_task)
        
        l_data = json.loads(llama_res.choices[0].message.content)
        m_data = {"text": mistral_res.choices[0].message.content}

        # 3. Raffinement DeepSeek & Sourcing
        ds_json = json.loads(await deepseek_refine(l_data, m_data))
        final_links = await search_perplexity(ds_json.get('search_query'))

        return f"""
        <div class="results animate-in">
            <div class="res-card mat"><strong>🧪 Matière (Consensus)</strong><p>{l_data.get('mat', 'Analyse en cours')}</p></div>
            <div class="res-card std"><strong>📏 Standard Technique</strong><p>{l_data.get('std', 'Standard non défini')}</p></div>
            <div class="res-card shop"><strong>🛒 Liens Directs (Vérifiés)</strong>
                <p style="font-size:0.7rem; color:grey">Recherche : {ds_json.get('search_query')}</p>
                <div class="links-list">{final_links}</div>
            </div>
            {f'<div class="box-warn">❓ {ds_json["check"]}</div>' if ds_json.get('check') else ''}
            <button class="btn btn-run" onclick="location.reload()">🔄 Nouveau Scan</button>
        </div>
        """
    except Exception as e:
        return f"<div class='res-card' style='color:red'>Erreur Système : {str(e)}</div>"

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
        <title>PartFinder PRO</title>
        <style>
            :root { --p: #ea580c; --b: #0f172a; }
            body { font-family: -apple-system, sans-serif; background: #f1f5f9; padding: 15px; margin: 0; }
            .container { max-width: 450px; margin: auto; background: white; border-radius: 20px; padding: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
            h1 { color: var(--p); text-align: center; margin-bottom: 25px; font-size: 1.5rem; }
            .btn { width: 100%; padding: 18px; border-radius: 12px; border: none; font-weight: bold; cursor: pointer; font-size: 1rem; margin-bottom: 12px; }
            .btn-cam { background: var(--p); color: white; }
            .btn-run { background: var(--b); color: white; }
            #preview { width: 100%; border-radius: 12px; margin-bottom: 15px; display: none; }
            .input-box { position: relative; margin-bottom: 15px; }
            textarea { width: 100%; padding: 15px; border: 1px solid #ddd; border-radius: 12px; box-sizing: border-box; min-height: 70px; width:100%; }
            .mic-btn { position: absolute; right: 10px; top: 12px; font-size: 1.5rem; background: none; border: none; cursor: pointer; }
            .res-card { padding: 15px; border-radius: 12px; margin-top: 15px; border: 1px solid #e2e8f0; }
            .mat { border-left: 5px solid var(--p); }
            .std { border-left: 5px solid #10b981; }
            .shop { border-left: 5px solid #3b82f6; background: #f0f9ff; }
            .buy-link { display: block; background: white; border: 1.5px solid #3b82f6; color: #3b82f6; padding: 12px; border-radius: 10px; text-decoration: none; font-weight: bold; margin: 10px 0; text-align: center; }
            .box-warn { background: #fff7ed; border: 1px solid #ffedd5; padding: 10px; border-radius: 10px; color: #9a3412; font-size: 0.9rem; margin-top:10px; }
            #loader { display: none; text-align: center; color: var(--p); font-weight: bold; padding: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PartFinder PRO</h1>
            <button class="btn btn-cam" onclick="document.getElementById('f').click()">📸 PHOTOGRAPHIER</button>
            <input type="file" id="f" accept="image/*" capture="environment" hidden onchange="pv(this)">
            <img id="preview">
            <div class="input-box">
                <textarea id="ctx" placeholder="Où est située la pièce ?"></textarea>
                <button class="mic-btn" onclick="dictate()">🎙️</button>
            </div>
            <button id="go" class="btn btn-run" onclick="run()">LANCER LE DIAGNOSTIC</button>
            <div id="loader">⚙️ Triple Analyse (Llama + Mistral + DeepSeek)...</div>
            <div id="res"></div>
        </div>
        <script>
            let img;
            window.onload = () => { const last = localStorage.getItem('last_scan'); if(last) document.getElementById('res').innerHTML = last; };
            function pv(i) { img = i.files[0]; const r = new FileReader(); r.onload = (e) => { const p = document.getElementById('preview'); p.src = e.target.result; p.style.display = 'block'; }; r.readAsDataURL(img); }
            function dictate() { const sr = new (window.SpeechRecognition || window.webkitSpeechRecognition)(); sr.lang = 'fr-FR'; sr.onresult = (e) => { document.getElementById('ctx').value = e.results[0][0].transcript; }; sr.start(); }
            async function run() {
                if(!img) return alert("Photo requise");
                document.getElementById('loader').style.display="block";
                document.getElementById('go').style.display="none";
                const fd = new FormData(); fd.append('image', img); fd.append('context', document.getElementById('ctx').value);
                try {
                    const r = await fetch('/identify', { method: 'POST', body: fd });
                    const html = await r.text();
                    document.getElementById('res').innerHTML = html;
                    localStorage.setItem('last_scan', html);
                } catch (e) { alert("Erreur connexion"); } finally { document.getElementById('loader').style.display="none"; }
            }
        </script>
    </body>
    </html>
    """
