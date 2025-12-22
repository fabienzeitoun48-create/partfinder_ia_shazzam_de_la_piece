import os
import base64
import httpx
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

# Imports des agents (doivent être à la racine avec app.py)
import database_standards as db
from agent_expert_matiere import agent_expert_matiere
from agent_standardiste import agent_standardiste
from agent_sourcer import agent_sourcer

load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

async def search_perplexity(query: str):
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key: return "Recherche indisponible."
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": query}]
    }
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, json=data, headers=headers, timeout=20.0)
            return res.json()['choices'][0]['message']['content']
        except: return "Erreur de connexion au web."

@app.post("/identify")
async def identify(image: UploadFile = File(...), context: str = Form("")):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    img_bytes = await image.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # 1. Analyse Vision via Groq
    vision_res = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": f"Identifie cette pièce. Contexte : {context}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]}]
    ).choices[0].message.content

    # 2. Orchestration des agents
    matiere = agent_expert_matiere(vision_res)
    standard = agent_standardiste(vision_res)
    query_web = await agent_sourcer(vision_res)
    liens = await search_perplexity(query_web)

    # Rendu HTML final
    html_content = f"""
    <div class='section'><h3>🔧 Expertise Matériau</h3><p>{matiere}</p></div>
    <div class='section'><h3>📏 Standard Probable</h3><p>{standard}</p></div>
    <div class='section web'><h3>🛒 Où acheter ?</h3><p>{liens.replace('$', '&#36;')}</p></div>
    """
    return HTMLResponse(content=html_content)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: sans-serif; background: #f3f4f6; padding: 15px; }
            .card { background: white; max-width: 500px; margin: auto; padding: 20px; border-radius: 15px; border-top: 8px solid #ea580c; }
            h1 { color: #ea580c; text-align: center; }
            .btn { width: 100%; padding: 15px; border-radius: 8px; border: none; font-weight: bold; cursor: pointer; margin-top: 10px; }
            .btn-main { background: #ea580c; color: white; }
            .section { background: #fff7ed; border-left: 4px solid #ea580c; padding: 10px; margin-top: 10px; border-radius: 4px; }
            .web { background: #eff6ff; border-left-color: #2563eb; }
            #preview { width: 100%; border-radius: 10px; margin-top: 10px; display: none; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🔍 PartFinder AI</h1>
            <img id="preview">
            <button class="btn" style="background:#6b7280; color:white;" onclick="document.getElementById('in').click()">📸 PHOTO DE LA PIÈCE</button>
            <input type="file" id="in" accept="image/*" capture="environment" hidden onchange="pv(this)">
            <textarea id="ctx" style="width:100%; margin-top:10px; height:60px;" placeholder="Détails (ex: sous l'évier)"></textarea>
            <button id="go" class="btn btn-main" onclick="run()">LANCER LE SHAZAM</button>
            <div id="loading" style="display:none; text-align:center;">Analyse en cours...</div>
            <div id="result"></div>
        </div>
        <script>
            let file;
            function pv(i) { file = i.files[0]; const r = new FileReader(); r.onload=(e)=>{ const p=document.getElementById('preview'); p.src=e.target.result; p.style.display='block'; }; r.readAsDataURL(file); }
            async function run() {
                const res = document.getElementById('result'); const load = document.getElementById('loading');
                res.innerHTML = ""; load.style.display = "block";
                const fd = new FormData(); fd.append('image', file); fd.append('context', document.getElementById('ctx').value);
                const r = await fetch('/identify', { method: 'POST', body: fd });
                res.innerHTML = await r.text(); load.style.display = "none";
            }
        </script>
    </body>
    </html>
    """
