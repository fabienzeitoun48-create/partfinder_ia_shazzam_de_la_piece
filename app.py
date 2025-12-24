import os
import base64
import httpx
import asyncio
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

# IMPORTS DIRECTS (Fichiers à la racine, pas de dossiers)
import database_standards as db
from agent_expert_matiere import agent_expert_matiere
from agent_standardiste import agent_standardiste
from agent_sourcer import agent_sourcer

load_dotenv()

app = FastAPI()

# Configuration CORS (Indispensable pour mobile/PWA)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FONCTIONS UTILITAIRES ---

async def search_perplexity_async(query: str):
    """Recherche web asynchrone pour ne pas bloquer le serveur."""
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key: return "Recherche désactivée (Clé manquante)."
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Temperature 0 pour éviter d'inventer des magasins
    data = {
        "model": "sonar-pro",
        "temperature": 0,
        "messages": [{"role": "user", "content": query}]
    }
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, json=data, headers=headers, timeout=20.0)
            return res.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Erreur de connexion sourcing : {str(e)}"

# Wrappers pour exécuter les agents synchrones en parallèle
def run_expert(vision_data): return agent_expert_matiere(vision_data)
def run_standard(vision_data): return agent_standardiste(vision_data)
def run_sourcer_query(vision_data): return agent_sourcer(vision_data)

# --- ENDPOINT PRINCIPAL ---

@app.post("/identify", response_class=HTMLResponse)
async def identify(image: UploadFile = File(...), context: str = Form("")):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key: return "<div style='color:red'>Erreur : API Key manquante</div>"

    client = Groq(api_key=api_key)
    
    try:
        # 1. PRÉPARATION IMAGE
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # 2. VISION STRICTE (Modèle Llama 4 Scout)
        # Temperature = 0 pour désactiver la créativité
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.0, 
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un expert technique industriel rigoureux. Analyse l'image factuellement. Si un détail (filetage, matière) n'est pas visible, dis 'Non visible'. N'invente rien."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Identifie cette pièce technique. Contexte : {context}. Analyse uniquement ce que tu vois."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ]
        )
        vision_res = completion.choices[0].message.content

        # 3. EXÉCUTION PARALLÈLE DES AGENTS
        # On lance les 3 agents en même temps sans attendre que le premier finisse
        loop = asyncio.get_running_loop()
        
        future_matiere = loop.run_in_executor(None, run_expert, vision_res)
        future_standard = loop.run_in_executor(None, run_standard, vision_res)
        future_query = loop.run_in_executor(None, run_sourcer_query, vision_res)

        # On attend tous les résultats simultanément (Gain de temps : ~50%)
        matiere, standard, query_web = await asyncio.gather(future_matiere, future_standard, future_query)

        # 4. SOURCING WEB (Se lance une fois la requête prête)
        liens = await search_perplexity_async(query_web)
        
        # 5. RENDU HTML
        return f"""
        <div class="result-box">
            <div class="card strict-analysis">
                <h3>🔧 Analyse Technique (Rigueur : Max)</h3>
                <p><strong>Matière détectée :</strong> {matiere}</p>
                <p><strong>Standard identifié :</strong> {standard}</p>
            </div>
            
            <div class="card sourcing-results">
                <h3>🛒 Disponibilité Réelle</h3>
                <div class="links">{liens.replace('$', '&#36;')}</div>
            </div>
            
            <details style="margin-top:10px; font-size:0.8em; color:#666;">
                <summary>Voir l'analyse brute de l'IA</summary>
                <p>{vision_res}</p>
            </details>
        </div>
        """

    except Exception as e:
        return f"<div style='color:red; border:1px solid red; padding:10px;'>Erreur système : {str(e)}</div>"

# --- FRONTEND (INTERFACE) ---

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
            :root { --main: #ea580c; --bg: #f8fafc; }
            body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); margin: 0; padding: 10px; color: #1e293b; }
            .app { max-width: 600px; margin: 0 auto; background: white; padding: 20px; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
            h1 { color: var(--main); text-align: center; font-size: 1.5rem; margin-top: 0; }
            
            .btn { display: block; width: 100%; padding: 14px; border: none; border-radius: 8px; font-weight: 600; font-size: 1rem; cursor: pointer; transition: 0.2s; margin-bottom: 10px; }
            .btn-cam { background: #334155; color: white; }
            .btn-run { background: var(--main); color: white; }
            .btn:active { transform: scale(0.98); }
            
            #preview { width: 100%; border-radius: 8px; margin: 10px 0; display: none; border: 2px solid #e2e8f0; }
            textarea { width: 100%; padding: 10px; border: 1px solid #cbd5e1; border-radius: 8px; box-sizing: border-box; font-family: inherit; margin-bottom: 10px; }
            
            .card { padding: 15px; border-radius: 8px; margin-bottom: 15px; }
            .strict-analysis { background: #fff7ed; border-left: 4px solid #ea580c; }
            .sourcing-results { background: #eff6ff; border-left: 4px solid #3b82f6; }
            h3 { margin-top: 0; font-size: 1.1rem; }
            
            .loader { border: 3px solid #f3f3f3; border-top: 3px solid var(--main); border-radius: 50%; width: 24px; height: 24px; animation: spin 1s linear infinite; margin: 20px auto; display: none; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="app">
            <h1>⚙️ PartFinder <span style="font-size:0.6em; color:#64748b; vertical-align:middle;">PRO</span></h1>
            
            <button class="btn btn-cam" onclick="document.getElementById('file').click()">📸 PRENDRE PHOTO</button>
            <input type="file" id="file" accept="image/*" capture="environment" hidden onchange="preview(this)">
            
            <img id="preview">
            <textarea id="ctx" placeholder="Contexte technique (ex: raccord haute pression)..."></textarea>
            
            <button id="go" class="btn btn-run" onclick="analyze()">LANCER ANALYSE</button>
            
            <div id="loader" class="loader"></div>
            <div id="result"></div>
        </div>

        <script>
            let f;
            function preview(i) {
                f = i.files[0];
                const r = new FileReader();
                r.onload = (e) => {
                    const p = document.getElementById('preview');
                    p.src = e.target.result;
                    p.style.display = 'block';
                };
                r.readAsDataURL(f);
            }

            async function analyze() {
                if(!f) return alert("Photo requise");
                const res = document.getElementById('result');
                const load = document.getElementById('loader');
                const btn = document.getElementById('go');
                
                res.innerHTML = "";
                load.style.display = "block";
                btn.disabled = true;
                btn.style.opacity = 0.7;

                const fd = new FormData();
                fd.append('image', f);
                fd.append('context', document.getElementById('ctx').value);

                try {
                    const r = await fetch('/identify', { method: 'POST', body: fd });
                    if (r.ok) res.innerHTML = await r.text();
                    else res.innerHTML = "<p style='color:red'>Erreur serveur.</p>";
                } catch (e) {
                    res.innerHTML = "<p style='color:red'>Problème de connexion.</p>";
                } finally {
                    load.style.display = "none";
                    btn.disabled = false;
                    btn.style.opacity = 1;
                }
            }
        </script>
    </body>
    </html>
    """
