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
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --- RECHERCHE ULTRA-PRÉCISE (FICHES PRODUITS) ---
async def search_perplexity_async(query: str):
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key: return "⚠️ API Key manquante."
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    # On force l'IA à trouver des liens de fiches produits réelles
    data = {
        "model": "sonar", 
        "messages": [
            {"role": "system", "content": "Tu es un expert en pièces détachées. Ta mission : trouver des liens de fiches produits réelles (Deep Links) pour l'objet exact. Ne donne pas de liens vers des pages d'accueil ou des catégories générales. Donne 3 liens directs : Nom de l'article - Prix - [Cliquer ici](URL)."},
            {"role": "user", "content": f"Trouve la fiche produit exacte pour : {query} sur des sites comme RS Components, ManoMano, Amazon, ou Leroy Merlin."}
        ],
        "temperature": 0
    }
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, json=data, headers=headers, timeout=25.0)
            if res.status_code != 200: return f"Erreur sourcing : {res.status_code}"
            
            content = res.json()['choices'][0]['message']['content']
            # Nettoyage et transformation en boutons cliquables
            html_links = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank" class="buy-link">🔍 Voir le produit</a>', content)
            return html_links.replace("\n", "<br>")
        except:
            return "Recherche expirée. Réessayez."

@app.post("/identify", response_class=HTMLResponse)
async def identify(image: UploadFile = File(...), context: str = Form("")):
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    
    try:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # On demande à Llama Scout d'être extrêmement descriptif pour la recherche web
        prompt = """
        IDENTIFICATION TECHNIQUE POUR ACHAT.
        Réponds UNIQUEMENT en JSON :
        {
          "matiere": "Détails visuels précis (ex: Laiton chromé, traces de calcaire, poignée levier)",
          "standard": "Dimensions et type exact (ex: Mélangeur évier, entraxe 150mm, filetage 1/2)",
          "recherche_exacte": "Nom ultra-précis pour recherche Google Shopping (ex: Mitigeur évier bec fondu Grohe BauEdge 31367000)"
        }
        """
        
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [{"type": "text", "text": f"{prompt} Contexte: {context}"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        data = json.loads(completion.choices[0].message.content)
        # On utilise 'recherche_exacte' pour que Perplexity ne cherche pas n'importe quoi
        links = await search_perplexity_async(data.get("recherche_exacte"))
        
        return f"""
        <div class="results animate-in">
            <div class="res-card mat"><strong>🧪 Diagnostic Matériau</strong><p>{data.get('matiere')}</p></div>
            <div class="res-card std"><strong>📏 Caractéristiques</strong><p>{data.get('standard')}</p></div>
            <div class="res-card shop"><strong>🛒 Fiches Produits Trouvées</strong><div class="links-list">{links}</div></div>
            <button class="btn btn-run" onclick="location.reload()">🔄 Nouveau Scan</button>
        </div>
        """
    except Exception as e:
        return f"<div class='res-card' style='color:red'>Erreur : {str(e)}</div>"

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
            :root { --p: #ea580c; --b: #1e293b; }
            body { font-family: -apple-system, sans-serif; background: #f8fafc; padding: 15px; margin: 0; }
            .container { max-width: 480px; margin: auto; background: white; border-radius: 24px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); }
            h1 { color: var(--p); text-align: center; margin-bottom: 5px; font-size: 1.6rem; }
            p.info { text-align: center; color: #64748b; font-size: 0.85rem; margin-bottom: 25px; }
            .btn { width: 100%; padding: 18px; border-radius: 14px; border: none; font-weight: bold; cursor: pointer; font-size: 1rem; margin-bottom: 12px; transition: 0.2s; }
            .btn-cam { background: var(--p); color: white; box-shadow: 0 4px 15px rgba(234, 88, 12, 0.2); }
            .btn-run { background: var(--b); color: white; }
            #preview { width: 100%; border-radius: 16px; margin-bottom: 15px; display: none; }
            
            .input-area { position: relative; margin-bottom: 15px; }
            textarea { width: 100%; padding: 15px; border: 1.5px solid #e2e8f0; border-radius: 14px; box-sizing: border-box; font-family: inherit; resize: none; min-height: 80px; }
            .mic-btn { position: absolute; right: 12px; top: 12px; font-size: 1.6rem; background: none; border: none; cursor: pointer; }
            
            .res-card { padding: 18px; border-radius: 14px; margin-top: 15px; background: #fff; border: 1px solid #e2e8f0; }
            .mat { border-left: 5px solid var(--p); }
            .std { border-left: 5px solid #10b981; }
            .shop { border-left: 5px solid #3b82f6; background: #f0f9ff; }
            
            .buy-link { display: block; background: white; border: 1.5px solid #3b82f6; color: #3b82f6; padding: 12px; border-radius: 10px; text-decoration: none; font-weight: bold; margin: 10px 0; text-align: center; font-size: 0.95rem; }
            .buy-link:active { background: #3b82f6; color: white; }
            
            #loader { display: none; text-align: center; color: var(--p); font-weight: bold; padding: 30px; }
            .animate-in { animation: fadeIn 0.4s ease-out; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PartFinder PRO</h1>
            <p class="info">Identification technique & Deep-sourcing</p>
            
            <button class="btn btn-cam" onclick="document.getElementById('f').click()">📸 Scanner la pièce</button>
            <input type="file" id="f" accept="image/*" capture="environment" hidden onchange="pv(this)">
            <img id="preview">
            
            <div class="input-area">
                <textarea id="ctx" placeholder="Décrivez l'emplacement ou la panne..."></textarea>
                <button class="mic-btn" onclick="dictate()">🎙️</button>
            </div>
            
            <button id="go" class="btn btn-run" onclick="run()">Lancer l'Analyse</button>
            <div id="loader">⚙️ Recherche de références exactes...</div>
            <div id="res"></div>
        </div>

        <script>
            let img;
            // Sauvegarde locale pour éviter de tout perdre au refresh
            window.onload = () => {
                const saved = localStorage.getItem('last_res');
                if(saved) document.getElementById('res').innerHTML = saved;
            };

            function pv(i) {
                img = i.files[0];
                const r = new FileReader();
                r.onload = (e) => { 
                    const p = document.getElementById('preview');
                    p.src = e.target.result; p.style.display = 'block';
                };
                r.readAsDataURL(img);
            }

            function dictate() {
                const sr = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                sr.lang = 'fr-FR';
                sr.onresult = (e) => { document.getElementById('ctx').value = e.results[0][0].transcript; };
                sr.start();
            }

            async function run() {
                if(!img) return alert("Photo requise");
                document.getElementById('loader').style.display="block";
                document.getElementById('res').innerHTML = "";
                document.getElementById('go').style.display="none";
                
                const fd = new FormData();
                fd.append('image', img);
                fd.append('context', document.getElementById('ctx').value);
                
                try {
                    const r = await fetch('/identify', { method: 'POST', body: fd });
                    const html = await r.text();
                    document.getElementById('res').innerHTML = html;
                    localStorage.setItem('last_res', html); // Sauvegarde
                } catch (e) {
                    alert("Erreur de connexion");
                } finally {
                    document.getElementById('loader').style.display="none";
                }
            }
        </script>
    </body>
    </html>
    """
