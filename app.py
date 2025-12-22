import os
import base64
import httpx
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

# Imports des agents - Fichiers situés à la racine sur GitHub
import database_standards as db
from agent_expert_matiere import agent_expert_matiere
from agent_standardiste import agent_standardiste
from agent_sourcer import agent_sourcer

load_dotenv()

app = FastAPI()

# Configuration pour autoriser les accès mobiles et PWA
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def search_perplexity(query: str):
    """Recherche web temps réel via Perplexity."""
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        return "Recherche indisponible (Clé manquante)."
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": query}]
    }
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, json=data, headers=headers, timeout=25.0)
            return res.json()['choices'][0]['message']['content']
        except Exception:
            return "Impossible de récupérer les liens marchands."

@app.post("/identify", response_class=HTMLResponse)
async def identify(image: UploadFile = File(...), context: str = Form("")):
    """Diagnostic visuel utilisant Llama 4 Scout."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "<p style='color:red;'>Erreur : Variable GROQ_API_KEY absente sur Render.</p>"

    client = Groq(api_key=api_key)
    
    try:
        # Encodage de l'image pour l'IA
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # 1. Analyse Vision via Groq avec le modèle Llama 4 Scout
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyse cette pièce détachée. Contexte : {context}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ]
        )
        vision_analysis = completion.choices[0].message.content

        # 2. Traitement par les agents métier
        matiere = agent_expert_matiere(vision_analysis)
        standard = agent_standardiste(vision_analysis)
        
        # 3. Recherche de disponibilité
        query_web = await agent_sourcer(vision_analysis)
        liens = await search_perplexity(query_web)

        # 4. Rendu HTML formaté pour l'application
        return f"""
        <div style="background:white; border-radius:10px;">
            <div style="border-left:5px solid #ea580c; background:#fff7ed; padding:15px; margin-bottom:15px; border-radius:8px;">
                <h3 style="color:#ea580c; margin-top:0;">🔧 Identification</h3>
                <p><strong>Matière :</strong> {matiere}</p>
                <p><strong>Standard :</strong> {standard}</p>
            </div>
            
            <div style="border-left:5px solid #2563eb; background:#eff6ff; padding:15px; border-radius:8px;">
                <h3 style="color:#2563eb; margin-top:0;">🛒 Disponibilité Web</h3>
                <div style="font-size:0.9em; line-height:1.4;">{liens.replace('$', '&#36;')}</div>
            </div>
        </div>
        """
    except Exception as e:
        return f"<div style='color:red; border:1px solid red; padding:10px;'>Erreur Groq : {str(e)}</div>"

@app.get("/", response_class=HTMLResponse)
def home():
    """Interface utilisateur de PartFinder AI."""
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PartFinder AI</title>
        <style>
            body { font-family: sans-serif; background: #f3f4f6; padding: 15px; margin: 0; }
            .container { max-width: 500px; margin: auto; background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-top: 8px solid #ea580c; }
            h1 { color: #ea580c; text-align: center; font-size: 1.5rem; }
            .btn { width: 100%; padding: 15px; border-radius: 8px; border: none; font-weight: bold; cursor: pointer; margin-top: 15px; font-size: 1rem; }
            .btn-camera { background: #6b7280; color: white; margin-bottom: 10px; }
            .btn-run { background: #ea580c; color: white; }
            #preview { width: 100%; border-radius: 8px; margin-top: 15px; display: none; }
            textarea { width: 100%; margin-top: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; box-sizing: border-box; }
            #loading { display: none; text-align: center; font-weight: bold; color: #ea580c; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 PartFinder AI</h1>
            <p style="text-align:center; font-size:0.9rem; color:#666;">Le Shazam des pièces détachées</p>
            
            <button class="btn btn-camera" onclick="document.getElementById('file').click()">📸 PHOTOGRAPHIER LA PIÈCE</button>
            <input type="file" id="file" accept="image/*" capture="environment" hidden onchange="showPv(this)">
            
            <img id="preview">
            
            <textarea id="ctx" placeholder="Ajoutez un contexte (ex: évier de cuisine, marque...)"></textarea>
            
            <button id="go" class="btn btn-run" onclick="analyze()">IDENTIFIER ET TROUVER</button>
            
            <div id="loading">Analyse en cours...</div>
            <div id="result" style="margin-top:20px;"></div>
        </div>

        <script>
            let img;
            function showPv(i) {
                img = i.files[0];
                const r = new FileReader();
                r.onload = (e) => {
                    const p = document.getElementById('preview');
                    p.src = e.target.result;
                    p.style.display = 'block';
                };
                r.readAsDataURL(img);
            }

            async function analyze() {
                if(!img) return alert("Prenez une photo d'abord");
                const res = document.getElementById('result');
                const load = document.getElementById('loading');
                res.innerHTML = "";
                load.style.display = "block";

                const fd = new FormData();
                fd.append('image', img);
                fd.append('context', document.getElementById('ctx').value);

                try {
                    const r = await fetch('/identify', { method: 'POST', body: fd });
                    res.innerHTML = await r.text();
                } catch (e) {
                    res.innerHTML = "Erreur de connexion au serveur.";
                } finally {
                    load.style.display = "none";
                }
            }
        </script>
    </body>
    </html>
    """
