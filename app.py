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
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION SOURCING (CORRECTION ERREUR) ---
async def search_perplexity_async(query: str):
    """Recherche les vendeurs. Utilise le modèle stable."""
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key: return "⚠️ Clé API Perplexity manquante."
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Changement de modèle vers une version très stable
    data = {
        "model": "llama-3.1-sonar-large-online", 
        "temperature": 0.1, # Très faible pour éviter les hallucinations
        "messages": [
            {
                "role": "system", 
                "content": "Tu es un assistant d'achat B2B. Trouve 3 liens directs d'achat pour cette pièce exacte. Format: Nom du site - Prix (si dispo) - Lien court. Pas de blabla."
            },
            {
                "role": "user", 
                "content": f"Trouve où acheter cette pièce : {query}. Cherche sur Manomano, Amazon, RS Components, Cedeo."
            }
        ]
    }
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, json=data, headers=headers, timeout=30.0)
            if res.status_code != 200:
                return f"⚠️ Erreur API ({res.status_code}) : {res.text}"
            
            content = res.json()['choices'][0]['message']['content']
            # Formatage HTML des liens pour qu'ils soient cliquables et propres
            formatted = content.replace("- ", "<li>").replace("\n", "</li>")
            return f"<ul style='margin:0; padding-left:20px;'>{formatted}</ul>"
            
        except Exception as e:
            return f"⚠️ Erreur technique Sourcing : {str(e)}"

# --- ENDPOINT ANALYSE ---
@app.post("/identify", response_class=HTMLResponse)
async def identify(image: UploadFile = File(...), context: str = Form("")):
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    
    try:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # --- ETAPE 1 : ANALYSE UNIQUE & STRUCTURÉE (JSON) ---
        # Au lieu de 3 appels qui se répètent, on fait 1 appel JSON strict.
        # Cela garantit 0% de redondance entre matière et standard.
        
        system_prompt = """
        Tu es un scanner industriel de haute précision.
        Analyse l'image et renvoie UNIQUEMENT un objet JSON strict.
        Ne fais AUCUNE phrase. Sois chirurgical.
        Format attendu :
        {
            "matiere": "Nom précis (ex: Laiton Chrome)",
            "matiere_details": "2-3 mots clés (ex: Finition brossée, Traces oxydation)",
            "standard": "Type (ex: Filetage Gaz BSP, Raccord à compression)",
            "dimensions": "Estimation (ex: 15/21, Diamètre 12mm)",
            "search_query": "Requete de recherche web pour trouver cette pièce (ex: raccord laiton 15/21 male femelle achat)"
        }
        """
        
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview", # Modèle vision rapide
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyse cette pièce. Contexte : {context}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ],
            temperature=0,
            response_format={"type": "json_object"} # Force le JSON
        )
        
        # Parsing du résultat JSON
        result_json = json.loads(completion.choices[0].message.content)
        
        # --- ETAPE 2 : SOURCING (En parallèle du reste si besoin, mais ici séquentiel rapide) ---
        liens_achat = await search_perplexity_async(result_json.get("search_query", context))
        
        # --- ETAPE 3 : RENDU DASHBOARD MODERNE ---
        return f"""
        <div class="dashboard">
            <div class="widget">
                <div class="icon-header">
                    <span class="icon">🧪</span>
                    <span class="title">MATÉRIAU</span>
                </div>
                <div class="content highlight">
                    {result_json.get('matiere', 'Non identifié')}
                </div>
                <div class="details">
                    {result_json.get('matiere_details', '')}
                </div>
            </div>

            <div class="widget">
                <div class="icon-header">
                    <span class="icon">📏</span>
                    <span class="title">STANDARD</span>
                </div>
                <div class="content highlight">
                    {result_json.get('dimensions', '?')}
                </div>
                <div class="details">
                    {result_json.get('standard', '')}
                </div>
            </div>

            <div class="widget full-width sourcing-widget">
                <div class="icon-header">
                    <span class="icon">🛒</span>
                    <span class="title">ACHETER MAINTENANT</span>
                </div>
                <div class="links-container">
                    {liens_achat.replace('$', '&#36;')}
                </div>
            </div>

            <button class="btn-reset" onclick="resetApp()">🔄 NOUVEAU SCAN</button>
        </div>
        """

    except Exception as e:
        return f"<div class='error-box'>❌ ERREUR SYSTÈME : {str(e)}</div>"

# --- FRONTEND ---
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>PartFinder PRO</title>
        <style>
            :root { 
                --primary: #2563eb; 
                --bg: #f8fafc; 
                --card-bg: #ffffff;
                --text: #1e293b;
            }
            body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: var(--bg); margin: 0; padding: 15px; color: var(--text); -webkit-font-smoothing: antialiased; }
            
            /* Container Principal */
            .app-container { max-width: 480px; margin: 0 auto; }
            
            /* Header */
            .header { text-align: center; margin-bottom: 20px; }
            h1 { color: var(--primary); margin: 0; font-size: 1.5rem; letter-spacing: -0.5px; }
            .subtitle { color: #64748b; font-size: 0.85rem; margin-top: 5px; }

            /* Upload UI */
            .camera-btn {
                background: linear-gradient(135deg, #2563eb, #1d4ed8);
                color: white;
                border: none;
                border-radius: 16px;
                padding: 20px;
                width: 100%;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                transition: transform 0.1s;
            }
            .camera-btn:active { transform: scale(0.98); }
            
            #preview { width: 100%; border-radius: 12px; margin-top: 15px; display: none; border: 2px solid #e2e8f0; }

            /* Inputs */
            .input-zone { margin-top: 15px; position: relative; }
            textarea { 
                width: 100%; 
                padding: 12px; 
                border: 1px solid #e2e8f0; 
                border-radius: 12px; 
                font-family: inherit; 
                height: 50px; 
                box-sizing: border-box; 
                resize: none;
                background: white;
            }
            .mic-icon { position: absolute; right: 10px; top: 12px; font-size: 1.2rem; cursor: pointer; opacity: 0.6; }
            
            .analyze-btn {
                background: #0f172a;
                color: white;
                width: 100%;
                padding: 15px;
                border-radius: 12px;
                border: none;
                font-weight: bold;
                margin-top: 10px;
                font-size: 1rem;
            }

            /* --- RESULTATS DASHBOARD (CSS INJECTÉ) --- */
            .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 20px; }
            .widget { background: white; padding: 15px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.03); }
            .full-width { grid-column: span 2; }
            
            .icon-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; opacity: 0.7; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; }
            .content.highlight { font-size: 1.1rem; font-weight: 800; color: #0f172a; line-height: 1.2; }
            .details { font-size: 0.85rem; color: #64748b; margin-top: 4px; }
            
            .sourcing-widget { border-left: 4px solid #2563eb; background: #eff6ff; }
            .links-container ul { list-style: none; padding: 0; margin: 0; }
            .links-container li { margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #dbeafe; font-size: 0.9rem; }
            .links-container li:last-child { border: none; margin: 0; }
            
            .btn-reset { grid-column: span 2; background: transparent; border: 1px solid #cbd5e1; color: #64748b; padding: 10px; border-radius: 8px; margin-top: 10px; cursor: pointer; }
            
            .error-box { background: #fef2f2; color: #991b1b; padding: 15px; border-radius: 10px; border: 1px solid #fecaca; text-align: center; font-size: 0.9rem; }
            
            /* Loader */
            #loader { display: none; text-align: center; margin: 30px; }
            .spinner { width: 30px; height: 30px; border: 4px solid #e2e8f0; border-top-color: #2563eb; border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 10px; }
            @keyframes spin { to { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="app-container">
            <div class="header">
                <h1>PartFinder PRO</h1>
                <div class="subtitle">Scanner Industriel & Sourcing</div>
            </div>
            
            <div id="uploadSection">
                <button class="camera-btn" onclick="document.getElementById('file').click()">
                    <span>📸</span> Scanner une pièce
                </button>
                <input type="file" id="file" accept="image/*" capture="environment" hidden onchange="handleFile(this)">
            </div>

            <img id="preview">
            
            <div class="input-zone" id="inputZone">
                <textarea id="ctx" placeholder="Contexte (ex: raccord chaudière gaz)..."></textarea>
                <span class="mic-icon" onclick="startDictation()">🎙️</span>
            </div>
            
            <button id="goBtn" class="analyze-btn" onclick="runAnalysis()">LANCER L'ANALYSE</button>
            
            <div id="loader">
                <div class="spinner"></div>
                <span style="font-size:0.9rem; color:#64748b;">Identification & Recherche marchande...</span>
            </div>
            
            <div id="result"></div>
        </div>

        <script>
            let currentFile;

            function handleFile(input) {
                currentFile = input.files[0];
                if (currentFile) {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        const p = document.getElementById('preview');
                        p.src = e.target.result;
                        p.style.display = 'block';
                        document.getElementById('uploadSection').style.display = 'none';
                    };
                    reader.readAsDataURL(currentFile);
                }
            }

            function startDictation() {
                if (window.webkitSpeechRecognition) {
                    const r = new webkitSpeechRecognition();
                    r.lang = "fr-FR";
                    r.onresult = (e) => { document.getElementById('ctx').value += " " + e.results[0][0].transcript; };
                    r.start();
                } else { alert("Dictée non supportée"); }
            }

            async function runAnalysis() {
                if(!currentFile) return alert("Photo manquante !");
                
                const ui = {
                    res: document.getElementById('result'),
                    load: document.getElementById('loader'),
                    btn: document.getElementById('goBtn'),
                    inputs: document.getElementById('inputZone')
                };

                ui.res.innerHTML = "";
                ui.load.style.display = "block";
                ui.btn.style.display = "none";

                const fd = new FormData();
                fd.append('image', currentFile);
                fd.append('context', document.getElementById('ctx').value);

                try {
                    const response = await fetch('/identify', { method: 'POST', body: fd });
                    ui.res.innerHTML = await response.text();
                } catch (e) {
                    ui.res.innerHTML = "<div class='error-box'>Erreur de connexion au serveur.</div>";
                    ui.btn.style.display = "block";
                } finally {
                    ui.load.style.display = "none";
                }
            }

            window.resetApp = function() {
                currentFile = null;
                document.getElementById('file').value = "";
                document.getElementById('preview').style.display = 'none';
                document.getElementById('uploadSection').style.display = 'block';
                document.getElementById('ctx').value = "";
                document.getElementById('result').innerHTML = "";
                document.getElementById('goBtn').style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
