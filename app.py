import os
import base64
import httpx
import asyncio
import re
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

# Imports directs (Structure à plat sur GitHub)
import database_standards as db
from agent_expert_matiere import agent_expert_matiere
from agent_standardiste import agent_standardiste
from agent_sourcer import agent_sourcer

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FONCTION DE NETTOYAGE (ANTI-BLABLA) ---
def clean_response(text: str) -> str:
    """Supprime les phrases d'introduction de l'IA pour ne garder que le fait."""
    # Enlève les phrases type "En tant qu'expert...", "D'après l'image..."
    patterns = [
        r"^Tu es un.*?[.:]", 
        r"^Je suis un.*?[.:]",
        r"^En tant que.*?[,.:]",
        r"^D'après.*?[,.:]",
        r"^Voici.*?[.:]"
    ]
    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    return cleaned.strip()

# --- SOURCING ASYNCHRONE ---
async def search_perplexity_async(query: str):
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key: return "Indisponible."
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "sonar-pro",
        "temperature": 0,
        "messages": [{"role": "user", "content": query}]
    }
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, json=data, headers=headers, timeout=20.0)
            return res.json()['choices'][0]['message']['content']
        except: return "Erreur sourcing."

# --- WRAPPERS AGENTS ---
def run_expert(vision_data): return agent_expert_matiere(vision_data)
def run_standard(vision_data): return agent_standardiste(vision_data)
def run_sourcer_query(vision_data): return agent_sourcer(vision_data)

# --- ENDPOINT ANALYSE ---
@app.post("/identify", response_class=HTMLResponse)
async def identify(image: UploadFile = File(...), context: str = Form("")):
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    
    try:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # 1. VISION STRICTE & DISSOCIÉE
        # On demande explicitement de séparer physique (matière) et géométrie (standard)
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.0, 
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un scanner industriel. Analyse l'image. Sépare strictement : 1. L'aspect MATIÈRE (couleur, texture, oxydation). 2. L'aspect STANDARD (forme, filetage, type de tête). Sois factuel. Pas d'introduction."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyse cette pièce. Contexte: {context}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ]
        )
        vision_res = completion.choices[0].message.content

        # 2. PARALLÉLISATION
        loop = asyncio.get_running_loop()
        f_mat = loop.run_in_executor(None, run_expert, vision_res)
        f_std = loop.run_in_executor(None, run_standard, vision_res)
        f_qry = loop.run_in_executor(None, run_sourcer_query, vision_res)

        raw_matiere, raw_standard, query_web = await asyncio.gather(f_mat, f_std, f_qry)

        # 3. NETTOYAGE & SOURCING
        matiere = clean_response(raw_matiere)
        standard = clean_response(raw_standard)
        liens = await search_perplexity_async(query_web)
        
        return f"""
        <div class="result-box">
            <div class="card mat-card">
                <h3>🧪 Matière</h3>
                <p>{matiere}</p>
            </div>
            
            <div class="card std-card">
                <h3>📏 Standard & Dimensions</h3>
                <p>{standard}</p>
            </div>
            
            <div class="card buy-card">
                <h3>🛒 Où trouver ?</h3>
                <div class="links">{liens.replace('$', '&#36;')}</div>
            </div>

            <button class="btn btn-reset" onclick="resetApp()">🔄 NOUVEAU DIAGNOSTIC</button>
        </div>
        """

    except Exception as e:
        return f"<div class='error'>Erreur système : {str(e)}</div>"

# --- FRONTEND COMPLET ---
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
            :root { --p: #ea580c; --s: #3b82f6; --bg: #f1f5f9; }
            body { font-family: -apple-system, sans-serif; background: var(--bg); margin: 0; padding: 15px; color: #334155; }
            .container { max-width: 500px; margin: 0 auto; background: white; padding: 20px; border-radius: 20px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1); }
            
            h1 { color: var(--p); text-align: center; margin: 0 0 5px 0; font-size: 1.6rem; letter-spacing: -1px; }
            p.sub { text-align: center; color: #94a3b8; font-size: 0.9rem; margin-bottom: 25px; }
            
            /* Upload Zone */
            .upload-box { border: 2px dashed #cbd5e1; border-radius: 15px; padding: 30px; text-align: center; cursor: pointer; transition: 0.2s; background: #f8fafc; }
            .upload-box:active { border-color: var(--p); background: #fff7ed; }
            .cam-icon { font-size: 40px; display: block; margin-bottom: 10px; }
            
            #preview { width: 100%; border-radius: 12px; margin-top: 15px; display: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            
            /* Context & Mic */
            .input-group { position: relative; margin-top: 15px; }
            textarea { width: 100%; padding: 12px 45px 12px 12px; border: 1px solid #cbd5e1; border-radius: 10px; height: 60px; box-sizing: border-box; font-family: inherit; resize: none; }
            .mic-btn { position: absolute; right: 8px; top: 50%; transform: translateY(-50%); background: none; border: none; font-size: 20px; cursor: pointer; color: #64748b; padding: 5px; }
            .mic-btn.listening { color: var(--p); animation: pulse 1s infinite; }
            
            /* Actions */
            .btn { width: 100%; padding: 16px; border: none; border-radius: 12px; font-weight: 700; font-size: 1rem; margin-top: 15px; cursor: pointer; }
            .btn-run { background: var(--p); color: white; box-shadow: 0 4px 0 #c2410c; }
            .btn-run:active { transform: translateY(2px); box-shadow: none; }
            .btn-reset { background: #e2e8f0; color: #475569; margin-top: 20px; }
            
            /* Results */
            .card { padding: 15px; border-radius: 10px; margin-bottom: 12px; font-size: 0.95rem; line-height: 1.5; }
            .mat-card { background: #fff7ed; border-left: 4px solid var(--p); }
            .std-card { background: #f0fdf4; border-left: 4px solid #16a34a; }
            .buy-card { background: #eff6ff; border-left: 4px solid var(--s); }
            h3 { margin: 0 0 8px 0; font-size: 1rem; color: #1e293b; }
            
            /* Loader */
            #loader { display: none; text-align: center; margin: 20px; color: var(--p); font-weight: bold; }
            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        </style>
    </head>
    <body>
        <div class="container" id="mainApp">
            <h1>PartFinder PRO</h1>
            <p class="sub">Identification Industrielle Instantanée</p>
            
            <div class="upload-box" onclick="document.getElementById('file').click()" id="uploadZone">
                <span class="cam-icon">📸</span>
                <strong>Appuyez pour photographier</strong>
            </div>
            <input type="file" id="file" accept="image/*" capture="environment" hidden onchange="handleFile(this)">
            <img id="preview">
            
            <div class="input-group" id="ctxZone">
                <textarea id="ctx" placeholder="Dites ou écrivez le contexte (ex: raccord chaudière)..."></textarea>
                <button class="mic-btn" onclick="toggleMic()" title="Dicter">🎙️</button>
            </div>
            
            <button id="goBtn" class="btn btn-run" onclick="analyze()">LANCER L'IDENTIFICATION</button>
            
            <div id="loader">⚙️ Analyse technique en cours...</div>
            
            <div id="result"></div>
        </div>

        <script>
            let imgFile;
            
            // 1. Gestion Photo
            function handleFile(input) {
                imgFile = input.files[0];
                if (imgFile) {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        document.getElementById('preview').src = e.target.result;
                        document.getElementById('preview').style.display = 'block';
                        document.getElementById('uploadZone').style.display = 'none';
                    };
                    reader.readAsDataURL(imgFile);
                }
            }

            // 2. Gestion Micro (Speech-to-Text)
            function toggleMic() {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (!SpeechRecognition) return alert("Votre navigateur ne supporte pas la dictée.");
                
                const rec = new SpeechRecognition();
                rec.lang = 'fr-FR';
                const btn = document.querySelector('.mic-btn');
                
                rec.onstart = () => btn.classList.add('listening');
                rec.onend = () => btn.classList.remove('listening');
                
                rec.onresult = (event) => {
                    const text = event.results[0][0].transcript;
                    document.getElementById('ctx').value += (document.getElementById('ctx').value ? " " : "") + text;
                };
                rec.start();
            }

            // 3. Analyse Backend
            async function analyze() {
                if(!imgFile) return alert("Veuillez prendre une photo d'abord.");
                
                const ui = {
                    res: document.getElementById('result'),
                    load: document.getElementById('loader'),
                    btn: document.getElementById('goBtn'),
                    ctx: document.getElementById('ctxZone')
                };

                ui.res.innerHTML = "";
                ui.load.style.display = "block";
                ui.btn.style.display = "none"; // On cache le bouton pendant le chargement

                const fd = new FormData();
                fd.append('image', imgFile);
                fd.append('context', document.getElementById('ctx').value);

                try {
                    const r = await fetch('/identify', { method: 'POST', body: fd });
                    ui.res.innerHTML = await r.text();
                    // Le bouton Reset est inclus dans le HTML renvoyé par Python
                } catch (e) {
                    ui.res.innerHTML = "<p style='color:red;text-align:center'>Erreur de connexion.</p>";
                    ui.btn.style.display = "block"; // On remet le bouton si erreur
                } finally {
                    ui.load.style.display = "none";
                }
            }

            // 4. Reset / Nouveau Diagnostic
            window.resetApp = function() {
                imgFile = null;
                document.getElementById('file').value = "";
                document.getElementById('preview').style.display = 'none';
                document.getElementById('preview').src = "";
                document.getElementById('uploadZone').style.display = 'block';
                document.getElementById('ctx').value = "";
                document.getElementById('result').innerHTML = "";
                document.getElementById('goBtn').style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
