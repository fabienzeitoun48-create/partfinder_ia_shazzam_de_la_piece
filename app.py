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
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

async def search_perplexity_async(query: str):
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key: return "⚠️ Clé API absente."
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    # "sonar" est l'alias le plus stable pour éviter l'erreur 400
    data = {
        "model": "sonar", 
        "messages": [
            {"role": "system", "content": "Tu es un agent de sourcing. Trouve 3 sites marchands pour l'article précis cité. Donne Nom du site, Prix moyen et Lien direct. Sois précis et factuel."},
            {"role": "user", "content": f"Trouve des vendeurs pour : {query}"}
        ],
        "temperature": 0.2
    }
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, json=data, headers=headers, timeout=20.0)
            if res.status_code != 200:
                return f"Recherche indisponible (Erreur {res.status_code})"
            return res.json()['choices'][0]['message']['content'].replace("\n", "<br>")
        except:
            return "Délai de recherche dépassé."

@app.post("/identify", response_class=HTMLResponse)
async def identify(image: UploadFile = File(...), context: str = Form("")):
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    
    try:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # On demande une description technique concise mais complète
        prompt = """
        Analyse cette pièce industrielle.
        Réponds UNIQUEMENT en JSON avec ces 3 clés :
        1. "matiere": Description courte du matériau et de l'état (ex: Laiton poli avec traces d'usure).
        2. "standard": Analyse technique des formes et dimensions (ex: Crochet à visser, diamètre 4mm, filetage bois).
        3. "recherche": Terme exact pour achat (ex: Crochet Esse laiton 40mm).
        """
        
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
        links = await search_perplexity_async(data.get("recherche"))
        
        return f"""
        <div class="results">
            <div class="res-card mat">
                <strong>🧪 Matériau & État</strong>
                <p>{data.get('matiere')}</p>
            </div>
            <div class="res-card std">
                <strong>📐 Spécifications Techniques</strong>
                <p>{data.get('standard')}</p>
            </div>
            <div class="res-card shop">
                <strong>🛒 Points de vente trouvés</strong>
                <div class="links">{links}</div>
            </div>
            <button class="reset" onclick="location.reload()">🔄 Nouveau Scan</button>
        </div>
        """
    except Exception as e:
        return f"<div class='res-card' style='border-color:red'>Erreur : {str(e)}</div>"

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PartFinder AI</title>
        <style>
            body { font-family: -apple-system, sans-serif; background: #f0f2f5; padding: 15px; margin: 0; }
            .container { max-width: 480px; margin: auto; background: white; border-radius: 16px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
            h1 { color: #ea580c; text-align: center; font-size: 1.4rem; }
            .btn { width: 100%; padding: 16px; border-radius: 12px; border: none; font-weight: bold; cursor: pointer; font-size: 1rem; margin-bottom: 10px; }
            .btn-cam { background: #ea580c; color: white; }
            .btn-run { background: #1e293b; color: white; }
            #preview { width: 100%; border-radius: 12px; margin-bottom: 15px; display: none; }
            textarea { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 10px; box-sizing: border-box; }
            .res-card { padding: 15px; border-radius: 12px; margin-top: 10px; border: 1px solid #e5e7eb; background: #f9fafb; }
            .mat { border-left: 4px solid #ea580c; }
            .std { border-left: 4px solid #10b981; }
            .shop { border-left: 4px solid #3b82f6; background: #eff6ff; }
            .reset { width: 100%; padding: 12px; margin-top: 15px; border-radius: 8px; border: 1px solid #cbd5e1; cursor: pointer; }
            #loader { display: none; text-align: center; padding: 20px; color: #ea580c; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PartFinder AI</h1>
            <button class="btn btn-cam" onclick="document.getElementById('f').click()">📸 Prendre une photo</button>
            <input type="file" id="f" accept="image/*" capture="environment" hidden onchange="pv(this)">
            <img id="preview">
            <textarea id="ctx" placeholder="Où est située cette pièce ?"></textarea>
            <button id="go" class="btn btn-run" onclick="run()">Lancer l'Analyse</button>
            <div id="loader">Recherche en cours...</div>
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
            async function run() {
                if(!img) return alert("Photo requise");
                document.getElementById('loader').style.display="block";
                document.getElementById('go').style.display="none";
                const fd = new FormData();
                fd.append('image', img);
                fd.append('context', document.getElementById('ctx').value);
                const r = await fetch('/identify', { method: 'POST', body: fd });
                document.getElementById('res').innerHTML = await r.text();
                document.getElementById('loader').style.display="none";
            }
        </script>
    </body>
    </html>
    """
