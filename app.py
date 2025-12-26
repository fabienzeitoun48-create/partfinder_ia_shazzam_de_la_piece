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

# --- FILTRE DE SÉCURITÉ URL ---
def is_valid_product_link(url: str) -> bool:
    """Vérifie si le lien n'est pas une catégorie ou une page de recherche."""
    forbidden = ['/category', '/cat/', '/search', '/recherche', '/famille', '/resultats', 'filter=']
    return not any(word in url.lower() for word in forbidden)

# --- SOURCING AMPLIFIÉ (PERPLEXITY) ---
async def search_perplexity_async(query: str):
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key: return "⚠️ Clé API Perplexity manquante."
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    # TON PROMPT AMPLIFIÉ INTÉGRÉ ICI
    system_content = (
        "Tu es un Expert en Sourcing Industriel B2B/B2C et spécialiste du Deep-Linking. "
        "Ta mission : extraire des liens directs vers des fiches produits unitaires (SKU). "
        "INTERDICTION : liens de pages d'accueil, de catégories ou de résultats de recherche. "
        "FILTRAGE : Ne donne que des URLs pointant vers un produit précis. "
        "DIMENSIONS : Le produit DOIT respecter les mesures citées. "
        "DOUTE : Si les dimensions manquent (filetage, diamètre), pose une QUESTION précise à l'utilisateur. "
        "FORMAT : Nom Article (Marque + Modèle + Taille) - Prix - [Cliquer ici](URL). "
        "Zéro créativité, zéro blabla."
    )

    data = {
        "model": "sonar", 
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Trouve la fiche produit exacte pour : {query}"}
        ],
        "temperature": 0
    }
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, json=data, headers=headers, timeout=28.0)
            if res.status_code != 200: return f"Erreur API ({res.status_code})"
            
            content = res.json()['choices'][0]['message']['content']
            
            # Transformation des liens Markdown en boutons HTML + Filtrage de sécurité
            def link_replacer(match):
                label, link = match.groups()
                if is_valid_product_link(link):
                    return f'<a href="{link}" target="_blank" class="buy-link">🛒 Voir le produit ({label})</a>'
                return "" # On supprime le lien s'il est générique

            html_output = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link_replacer, content)
            return html_output.replace("\n", "<br>")
        except:
            return "Délai de recherche dépassé (Serveur occupé)."

# --- ANALYSE VISION (LLAMA 4 SCOUT) ---
@app.post("/identify", response_class=HTMLResponse)
async def identify(image: UploadFile = File(...), context: str = Form("")):
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    
    try:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        prompt = """
        IDENTIFICATION TECHNIQUE CHIRURGICALE.
        Réponds UNIQUEMENT en JSON :
        {
          "mat": "Matière et état (ex: Laiton nickelé, usé)",
          "std": "Dimensions et type (ex: Mitigeur évier, entraxe 150mm)",
          "search": "Terme de recherche ultra-précis pour achat direct"
        }
        """
        
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [{"type": "text", "text": f"{prompt} Contexte: {context}"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        data = json.loads(completion.choices[0].message.content)
        links = await search_perplexity_async(data.get("search"))
        
        return f"""
        <div class="results animate-in">
            <div class="res-card mat"><strong>🧪 Matière</strong><p>{data.get('mat')}</p></div>
            <div class="res-card std"><strong>📏 Technique</strong><p>{data.get('std')}</p></div>
            <div class="res-card shop"><strong>🔗 Fiches Produits Directes</strong><div class="links-list">{links}</div></div>
            <button class="btn btn-run" onclick="location.reload()">🔄 Nouveau Diagnostic</button>
        </div>
        """
    except Exception as e:
        return f"<div class='res-card' style='color:red'>Erreur : {str(e)}</div>"

# --- INTERFACE ---
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
            #preview { width: 100%; border-radius: 12px; margin-bottom: 15px; display: none; border: 1.5px solid #eee; }
            
            .input-box { position: relative; margin-bottom: 15px; }
            textarea { width: 100%; padding: 15px; border: 1px solid #ddd; border-radius: 12px; box-sizing: border-box; font-family: inherit; resize: none; min-height: 70px; }
            .mic-btn { position: absolute; right: 10px; top: 12px; font-size: 1.5rem; background: none; border: none; cursor: pointer; }
            
            .res-card { padding: 15px; border-radius: 12px; margin-top: 15px; background: #fff; border: 1px solid #e2e8f0; }
            .mat { border-left: 5px solid var(--p); }
            .std { border-left: 5px solid #10b981; }
            .shop { border-left: 5px solid #3b82f6; background: #f0f9ff; }
            
            .buy-link { display: block; background: white; border: 1.5px solid #3b82f6; color: #3b82f6; padding: 12px; border-radius: 10px; text-decoration: none; font-weight: bold; margin: 10px 0; text-align: center; }
            
            #loader { display: none; text-align: center; color: var(--p); font-weight: bold; padding: 20px; }
            .animate-in { animation: fadeInUp 0.3s ease-out; }
            @keyframes fadeInUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PartFinder PRO</h1>
            <button class="btn btn-cam" onclick="document.getElementById('f').click()">📸 PHOTOGRAPHIER LA PIÈCE</button>
            <input type="file" id="f" accept="image/*" capture="environment" hidden onchange="pv(this)">
            <img id="preview">
            
            <div class="input-box">
                <textarea id="ctx" placeholder="Contexte (ex: robinet fuyant)..."></textarea>
                <button class="mic-btn" onclick="dictate()">🎙️</button>
            </div>
            
            <button id="go" class="btn btn-run" onclick="run()">LANCER L'IDENTIFICATION</button>
            <div id="loader">⚙️ Analyse Llama Scout & Sourcing...</div>
            <div id="res"></div>
        </div>

        <script>
            let img;
            // Persistance des données (LocalStorage)
            window.onload = () => {
                const last = localStorage.getItem('last_scan');
                if(last) document.getElementById('res').innerHTML = last;
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
                document.getElementById('go').style.display="none";
                const fd = new FormData();
                fd.append('image', img);
                fd.append('context', document.getElementById('ctx').value);
                
                try {
                    const r = await fetch('/identify', { method: 'POST', body: fd });
                    const html = await r.text();
                    document.getElementById('res').innerHTML = html;
                    localStorage.setItem('last_scan', html); // Sauvegarde locale
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
