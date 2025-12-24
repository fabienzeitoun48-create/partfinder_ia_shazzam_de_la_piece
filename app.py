import os
import base64
import httpx
import asyncio
import json
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

# --- SOURCING PERPLEXITY (ROBUSTE) ---
async def search_perplexity_async(query: str):
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key: return "⚠️ Variable PERPLEXITY_API_KEY manquante."
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    data = {
        "model": "llama-3.1-sonar-large-online",
        "messages": [
            {"role": "system", "content": "Expert achat industriel. Donne 3 liens marchands précis. Format: Site - Prix - Lien court. Zéro phrase d'intro."},
            {"role": "user", "content": f"Trouver cette pièce : {query}"}
        ],
        "temperature": 0
    }
    
    async with httpx.AsyncClient() as client:
        try:
            # Timeout de 25s pour laisser le temps à la recherche web de finir
            res = await client.post(url, json=data, headers=headers, timeout=25.0)
            if res.status_code != 200:
                return f"Erreur Sourcing (Code {res.status_code})"
            return res.json()['choices'][0]['message']['content'].replace("\n", "<br>")
        except Exception as e:
            return f"Lien non récupéré : {str(e)}"

# --- ANALYSE VISION (LLAMA 4 SCOUT) ---
@app.post("/identify", response_class=HTMLResponse)
async def identify(image: UploadFile = File(...), context: str = Form("")):
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    
    try:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Prompt chirurgical pour 0% de répétition et 0% de blabla
        prompt = """
        IDENTIFICATION INDUSTRIELLE STRICTE.
        Réponds UNIQUEMENT via cet objet JSON :
        {
          "mat": "Matière (ex: Acier zingué)",
          "std": "Standard/Taille (ex: M8, Pas 1.25)",
          "web": "Terme de recherche optimisé"
        }
        Ne décris pas l'image. Pas de politesse.
        """
        
        # PASSAGE SUR LLAMA 4 SCOUT (PROD STABLE)
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt} Contexte: {context}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        data = json.loads(completion.choices[0].message.content)
        
        # Sourcing lancé avec le terme généré par l'analyse vision
        links = await search_perplexity_async(data.get("web", context))
        
        return f"""
        <div class="res-grid">
            <div class="box">
                <span class="lab">MATIÈRE</span>
                <div class="val">{data.get('mat')}</div>
            </div>
            <div class="box">
                <span class="lab">STANDARD</span>
                <div class="val">{data.get('std')}</div>
            </div>
            <div class="box full">
                <span class="lab">SOURCING MARCHAND</span>
                <div class="links">{links}</div>
            </div>
            <button class="reset-btn" onclick="resetApp()">🔄 NOUVEAU DIAGNOSTIC</button>
        </div>
        """
    except Exception as e:
        return f"<div class='box full' style='color:red'>Erreur Identification : {str(e)}</div>"

# --- INTERFACE PWA ---
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
            body { font-family: sans-serif; background: #f1f5f9; margin: 0; padding: 15px; }
            .card { max-width: 480px; margin: auto; background: white; padding: 25px; border-radius: 20px; box-shadow: 0 10px 15px rgba(0,0,0,0.05); }
            h1 { color: var(--p); text-align: center; margin-top:0; font-size: 1.5rem; }
            .main-btn { width: 100%; background: var(--p); color: white; border: none; padding: 18px; border-radius: 12px; font-weight: bold; cursor: pointer; font-size: 1rem; }
            #preview { width: 100%; border-radius: 12px; margin: 15px 0; display: none; }
            .input-box { position: relative; margin: 15px 0; }
            textarea { width: 100%; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px; box-sizing: border-box; height: 60px; font-family: inherit; }
            .mic { position: absolute; right: 10px; top: 12px; font-size: 1.3rem; cursor: pointer; }
            
            /* Grid Resultats */
            .res-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 20px; }
            .box { background: white; padding: 15px; border-radius: 12px; border: 1px solid #e2e8f0; }
            .full { grid-column: span 2; border-left: 5px solid #3b82f6; background: #eff6ff; }
            .lab { color: #64748b; font-size: 0.7rem; font-weight: bold; display: block; margin-bottom: 5px; }
            .val { color: var(--b); font-weight: 800; font-size: 1rem; }
            .links { font-size: 0.9rem; line-height: 1.5; color: #1e293b; }
            .reset-btn { grid-column: span 2; background: #334155; color: white; border: none; padding: 12px; border-radius: 10px; margin-top: 10px; cursor: pointer; font-weight: bold; }
            
            #loader { display: none; text-align: center; color: var(--p); padding: 20px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>PartFinder PRO</h1>
            <button id="cam" class="main-btn" onclick="document.getElementById('f').click()">📸 PHOTOGRAPHIER LA PIÈCE</button>
            <input type="file" id="f" accept="image/*" capture="environment" hidden onchange="pv(this)">
            
            <img id="preview">
            
            <div class="input-box">
                <textarea id="ctx" placeholder="Où se situe cette pièce ?"></textarea>
                <span class="mic" onclick="dict()">🎙️</span>
            </div>
            
            <button id="go" class="main-btn" style="background:var(--b)" onclick="run()">LANCER LE DIAGNOSTIC</button>
            
            <div id="loader">⚙️ Analyse Llama 4 Scout...</div>
            <div id="res"></div>
        </div>

        <script>
            let fileObj;
            function pv(i) {
                fileObj = i.files[0];
                const r = new FileReader();
                r.onload = (e) => { 
                    const p = document.getElementById('preview');
                    p.src = e.target.result; p.style.display = 'block';
                    document.getElementById('cam').style.display = 'none';
                };
                r.readAsDataURL(fileObj);
            }

            function dict() {
                const sr = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                sr.lang = 'fr-FR';
                sr.onresult = (e) => { document.getElementById('ctx').value = e.results[0][0].transcript; };
                sr.start();
            }

            async function run() {
                if(!fileObj) return alert("Photo manquante");
                document.getElementById('loader').style.display = "block";
                document.getElementById('go').style.display = "none";
                
                const fd = new FormData();
                fd.append('image', fileObj);
                fd.append('context', document.getElementById('ctx').value);

                try {
                    const r = await fetch('/identify', { method: 'POST', body: fd });
                    document.getElementById('res').innerHTML = await r.text();
                } catch (e) {
                    alert("Erreur de connexion");
                } finally {
                    document.getElementById('loader').style.display = "none";
                }
            }

            function resetApp() { location.reload(); }
        </script>
    </body>
    </html>
    """
