import os
import base64
import httpx
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

# Imports des agents (doivent être à la racine du projet sur GitHub)
import database_standards as db
from agent_expert_matiere import agent_expert_matiere
from agent_standardiste import agent_standardiste
from agent_sourcer import agent_sourcer

load_dotenv()

app = FastAPI()

# Configuration CORS pour autoriser les requêtes mobiles
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def search_perplexity(query: str):
    """Fonction de recherche web pour l'agent Sourcer."""
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        return "Clé Perplexity manquante. Recherche marchande indisponible."
    
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
            return "Erreur lors de la recherche des revendeurs."

@app.post("/identify", response_class=HTMLResponse)
async def identify(image: UploadFile = File(...), context: str = Form("")):
    """Endpoint principal de diagnostic visuel."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "<p style='color:red;'>Erreur : Clé GROQ_API_KEY manquante dans les variables d'environnement.</p>"

    client = Groq(api_key=api_key)
    
    try:
        # Lecture et encodage de l'image
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # 1. Analyse Vision via Groq (Modèle corrigé pour éviter l'erreur 400)
        # On utilise le modèle vision le plus stable disponible
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Identifie cette pièce de quincaillerie ou plomberie. Contexte : {context}. Analyse les dimensions et le matériau."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ]
        )
        vision_analysis = completion.choices[0].message.content

        # 2. Appel des agents spécialisés (logique métier)
        matiere_info = agent_expert_matiere(vision_analysis)
        standard_info = agent_standardiste(vision_analysis)
        
        # 3. Sourcing web via Perplexity
        sourcing_query = await agent_sourcer(vision_analysis)
        liens_achat = await search_perplexity(sourcing_query)

        # 4. Construction du rendu HTML propre pour l'application
        # Utilisation de entités HTML pour éviter les conflits avec le symbole $ de Perplexity
        liens_clean = liens_achat.replace('$', '&#36;')

        return f"""
        <div class="result-container">
            <div class="section-card" style="border-left: 5px solid #ea580c; background: #fff7ed; padding: 15px; margin-bottom: 15px; border-radius: 8px;">
                <h3 style="color: #ea580c; margin-top: 0;">🔧 Analyse de la Pièce</h3>
                <p><strong>Expertise Matériau :</strong> {matiere_info}</p>
                <p><strong>Standards détectés :</strong> {standard_info}</p>
            </div>
            
            <div class="section-card" style="border-left: 5px solid #2563eb; background: #eff6ff; padding: 15px; border-radius: 8px;">
                <h3 style="color: #2563eb; margin-top: 0;">🛒 Sourcing & Disponibilité</h3>
                <div style="font-size: 0.95em; line-height: 1.5;">{liens_clean}</div>
            </div>
        </div>
        """
    except Exception as e:
        return f"<div style='color:red; padding:20px; border:1px solid red;'>Erreur technique : {str(e)}</div>"

@app.get("/", response_class=HTMLResponse)
def home():
    """Page d'accueil de la PWA PartFinder."""
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>PartFinder AI</title>
        <style>
            :root { --primary: #ea580c; --secondary: #2563eb; --bg: #f3f4f6; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); margin: 0; padding: 15px; }
            .app-container { max-width: 500px; margin: auto; background: white; padding: 25px; border-radius: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); border-top: 10px solid var(--primary); }
            h1 { color: var(--primary); text-align: center; margin-bottom: 5px; font-size: 1.8rem; }
            p.subtitle { text-align: center; color: #6b7280; font-size: 0.9rem; margin-bottom: 25px; }
            .upload-zone { border: 2px dashed #d1d5db; border-radius: 12px; padding: 30px; text-align: center; cursor: pointer; transition: 0.3s; }
            .upload-zone:hover { border-color: var(--primary); background: #fff7ed; }
            #preview { width: 100%; border-radius: 12px; margin-top: 15px; display: none; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
            textarea { width: 100%; box-sizing: border-box; margin-top: 15px; padding: 12px; border: 1px solid #d1d5db; border-radius: 8px; font-family: inherit; height: 70px; }
            .btn { width: 100%; padding: 16px; border-radius: 10px; border: none; font-weight: bold; font-size: 1rem; cursor: pointer; margin-top: 15px; transition: 0.2s; }
            .btn-identify { background: var(--primary); color: white; box-shadow: 0 4px 0 #9a3412; }
            .btn-identify:active { transform: translateY(2px); box-shadow: 0 2px 0 #9a3412; }
            #loading { display: none; text-align: center; margin: 20px 0; font-weight: bold; color: var(--primary); }
            #result { margin-top: 25px; }
            .loader { border: 4px solid #f3f3f3; border-top: 4px solid var(--primary); border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 10px auto; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="app-container">
            <h1>🔍 PartFinder AI</h1>
            <p class="subtitle">Le Shazam de la pièce détachée</p>
            
            <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
                <span style="font-size: 40px;">📸</span><br>
                <strong>PHOTOGRAPHIER LA PIÈCE</strong>
                <p style="font-size: 0.8rem; color: #9ca3af;">Cliquez ici pour utiliser l'appareil photo</p>
            </div>
            
            <input type="file" id="fileInput" accept="image/*" capture="environment" hidden onchange="handleFile(this)">
            <img id="preview">
            
            <textarea id="contextInput" placeholder="Ex: C'est un raccord sous un évier de cuisine de 1995..."></textarea>
            
            <button id="submitBtn" class="btn btn-identify">IDENTIFIER ET TROUVER</button>
            
            <div id="loading">
                <div class="loader"></div>
                Identification en cours...
            </div>
            
            <div id="result"></div>
        </div>

        <script>
            let selectedFile;

            function handleFile(input) {
                selectedFile = input.files[0];
                if (selectedFile) {
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        const preview = document.getElementById('preview');
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    };
                    reader.readAsDataURL(selectedFile);
                }
            }

            document.getElementById('submitBtn').addEventListener('click', async () => {
                if (!selectedFile) {
                    alert("Veuillez d'abord prendre une photo.");
                    return;
                }

                const resultDiv = document.getElementById('result');
                const loadingDiv = document.getElementById('loading');
                const btn = document.getElementById('submitBtn');

                resultDiv.innerHTML = "";
                loadingDiv.style.display = "block";
                btn.disabled = true;
                btn.style.opacity = "0.5";

                const formData = new FormData();
                formData.append('image', selectedFile);
                formData.append('context', document.getElementById('contextInput').value);

                try {
                    const response = await fetch('/identify', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        resultDiv.innerHTML = await response.text();
                    } else {
                        resultDiv.innerHTML = "<p style='color:red;'>Erreur serveur lors de l'analyse.</p>";
                    }
                } catch (err) {
                    resultDiv.innerHTML = "<p style='color:red;'>Erreur réseau. Vérifiez votre connexion.</p>";
                } finally {
                    loadingDiv.style.display = "none";
                    btn.disabled = false;
                    btn.style.opacity = "1";
                }
            });
        </script>
    </body>
    </html>
    """
