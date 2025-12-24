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

# --- RECHERCHE ET NETTOYAGE DES LIENS ---
async def search_perplexity_async(query: str):
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key: return "⚠️ Clé API absente."
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    data = {
        "model": "sonar", 
        "messages": [
            {"role": "system", "content": "Trouve 3 sites marchands. Réponds UNIQUEMENT avec une liste Markdown : - [Nom du Site](URL) - Prix. Pas de texte avant ou après."},
            {"role": "user", "content": f"Acheter : {query}"}
        ],
        "temperature": 0
    }
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, json=data, headers=headers, timeout=25.0)
            if res.status_code != 200: return f"Recherche indisponible ({res.status_code})"
            
            content = res.json()['choices'][0]['message']['content']
            # Transformation du Markdown [Texte](URL) en HTML cliquable <a href='URL'>Texte</a>
            html_links = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank" class="buy-link">🔗 \1</a>', content)
            return html_links.replace("\n", "<br>")
        except:
            return "Délai dépassé."

@app.post("/identify", response_class=HTMLResponse)
async def identify(image: UploadFile = File(...), context: str = Form("")):
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    
    try:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        prompt = """
        Analyse technique :
        Réponds en JSON uniquement :
        {
          "matiere": "Description courte (ex: Laiton poli)",
          "standard": "Analyse technique (ex: Filetage gaz 1/2)",
          "recherche": "Nom pour achat"
        }
        """
        
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [{"type": "text", "text": f"{prompt} Contexte: {context}"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        data = json.loads(completion.choices[0].message.content)
        links = await search_perplexity_async(data.get("recherche"))
        
        return f"""
        <div class="results">
            <div class="res-card mat"><strong>🧪 Matériau</strong><p>{data.get('matiere')}</p></div>
            <div class="res-card std"><strong>📐 Technique</strong><p>{data.get('standard')}</p></div>
            <div class="res-card shop"><strong>🛒 Liens d'achat</strong><div class="links-list">{links}</div></div>
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
            body { font-family: -apple-system, sans-serif; background: #f4f4f9; padding: 15px; margin: 0; color: #333; }
            .container { max-width: 480px; margin: auto; background: white; border-radius: 20px; padding: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.1); }
            h1 { color: #ea580c; text-align: center; margin-bottom: 20px; }
            .btn { width: 100%; padding: 18px; border-radius: 12px; border: none; font-weight: bold; cursor: pointer; font-size: 1rem; margin-bottom: 10px; transition: 0.2s; }
            .btn-cam { background: #ea580c; color: white; }
            .btn-run { background: #1e293b; color: white; }
            #preview { width: 100%; border-radius: 15px; margin-bottom: 15px; display: none; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
            
            .input-area { position: relative; margin-bottom: 15px; }
            textarea { width: 100%; padding: 15px; border: 1px solid #ddd; border-radius: 12px; box-sizing: border-box; font-family: inherit; resize: none; font-size: 1rem; }
            .mic-btn { position: absolute; right: 10px; top: 12px; font-size: 1.5rem; cursor: pointer; background: none; border: none; }
            
            .res-card { padding: 15px; border-radius: 12px; margin-top: 12px; background: #fff; border: 1px solid #eee; }
            .mat { border-left: 5px solid #ea580c; }
            .std { border-left: 5px solid #10b981; }
            .shop { border-left: 5px solid #3b82f6; background: #f0f7ff; }
            
            /* Style des liens cliquables */
            .buy-link { display: inline-block; background: white; border: 1px solid #3b82f6; color: #3b82f6; padding: 8px 12px; border-radius: 8px; text-decoration: none; font-weight: bold; margin: 5px 0; font-size: 0.9rem; }
            .buy-link:hover { background: #3b82f6; color: white; }
            
            #loader { display: none; text-align: center; color: #ea580c; font-weight: bold; padding: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PartFinder PRO</h1>
            <button class="btn btn-cam" onclick="document.getElementById('f').click()">📸 Photo de la pièce</button>
            <input type="file" id="f" accept="image/*" capture="environment" hidden onchange="pv(this)">
            <img id="preview">
            
            <div class="input-area">
                <textarea id="ctx" placeholder="Où est située la pièce ?"></textarea>
                <button class="mic-btn" onclick="dictate()">🎙️</button>
            </div>
            
            <button id="go" class="btn btn-run" onclick="run()">Lancer l'Analyse</button>
            <div id="loader">⚙️ Analyse technique en cours...</div>
            <div id="res"></div>
        </div>

        <script>
            let img;
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
                if(!img) return alert("Prends une photo d'abord");
                document.getElementById('loader').style.display="block";
                document.getElementById('go').style.display="none";
                const fd = new FormData();
                fd.append('image', img);
                fd.append('context', document.getElementById('ctx').value);
                try {
                    const r = await fetch('/identify', { method: 'POST', body: fd });
                    document.getElementById('res').innerHTML = await r.text();
                } catch (e) {
                    alert("Erreur");
                } finally {
                    document.getElementById('loader').style.display="none";
                }
            }
        </script>
    </body>
    </html>
    """
