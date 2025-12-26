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
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

# --- LOGIQUE DE SOURCING ---
async def deepseek_consensus(llama_txt, mistral_txt, user_ctx):
    """DeepSeek agit en ingénieur méthode : il valide les matériaux et crée la requête d'achat."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    url = "https://api.deepseek.com/v1/chat/completions"
    
    # On injecte ton contexte utilisateur comme "Contrainte Critique"
    prompt = f"""
    CONTRAINTE UTILISATEUR : {user_ctx}
    RAPPORT VISION 1 : {llama_txt}
    RAPPORT VISION 2 : {mistral_txt}
    
    Mission : Générer un terme de recherche pour trouver la fiche produit EXACTE. 
    Si l'utilisateur dit INOX, le terme DOIT contenir 'Acier Inoxydable' ou 'Inox'.
    Format JSON STRICT : 
    {{"matiere": "verdict final", "technique": "dimensions/normes", "search": "terme shopping precis"}}
    """
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, 
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}},
                timeout=12.0)
            return json.loads(res.json()['choices'][0]['message']['content'])
        except:
            return {"matiere": "Erreur analyse", "technique": "Inconnue", "search": user_ctx}

async def search_perplexity_pro(query: str):
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    url = "https://api.perplexity.ai/chat/completions"
    
    # On force la recherche sur des sites de fournitures industrielles
    data = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Tu es un acheteur industriel. Trouve 3 URLs de fiches produits (RS, ManoMano, Leroy Merlin, Cedeo). Pas de pages d'accueil. Format: [Nom Produit](URL) - Prix."},
            {"role": "user", "content": f"Acheter immédiatement : {query}"}
        ]
    }
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=data, timeout=25.0)
            content = res.json()['choices'][0]['message']['content']
            # Nettoyage et transformation des liens en boutons cliquables
            html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank" class="buy-link">🛒 \1</a>', content)
            return html.replace("\n", "<br>")
        except:
            return "Aucun lien direct trouvé."

@app.post("/identify", response_class=HTMLResponse)
async def identify(image: UploadFile = File(...), context: str = Form("")):
    try:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        # Analyse Vision Parallèle
        l_task = asyncio.to_thread(groq_client.chat.completions.create,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Identifie : matière, type, filetage."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}],
            response_format={"type": "json_object"}
        )

        m_res = await mistral_client.chat.complete_async(
            model="pixtral-12b-2409",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Analyse technique précise de cet objet."}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}]}]
        )

        l_res = await l_task
        # Consensus par DeepSeek (Le cerveau logique)
        final_data = await deepseek_consensus(l_res.choices[0].message.content, m_res.choices[0].message.content, context)
        
        # Recherche finale basée sur le consensus
        links_html = await search_perplexity_pro(final_data['search'])

        return f"""
        <div class="results animate-in">
            <div class="res-card mat"><strong>🧪 Matériau validé</strong><p>{final_data['matiere']}</p></div>
            <div class="res-card std"><strong>📐 Fiche technique</strong><p>{final_data['technique']}</p></div>
            <div class="res-card shop"><strong>🔗 Liens d'achat directs</strong>
                <p style="font-size:0.75rem; color:#666">Cible : {final_data['search']}</p>
                <div class="links-list">{links_html}</div>
            </div>
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
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PartFinder Ultra</title>
        <style>
            body { font-family: sans-serif; background: #f0f4f8; padding: 20px; }
            .container { max-width: 450px; margin: auto; background: white; border-radius: 20px; padding: 25px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
            h1 { color: #ea580c; text-align: center; }
            .btn { width: 100%; padding: 18px; border-radius: 12px; border: none; font-weight: bold; cursor: pointer; margin-top: 10px; }
            .btn-cam { background: #ea580c; color: white; }
            .btn-run { background: #1e293b; color: white; }
            textarea { width: 100%; padding: 12px; border: 1px solid #ccc; border-radius: 10px; margin: 15px 0; box-sizing: border-box; }
            #preview { width: 100%; border-radius: 10px; display: none; margin-bottom: 10px; }
            .res-card { padding: 15px; border-radius: 10px; margin-top: 15px; border: 1px solid #ddd; background: #fff; }
            .mat { border-left: 5px solid #ea580c; }
            .std { border-left: 5px solid #10b981; }
            .shop { border-left: 5px solid #3b82f6; background: #f0f7ff; }
            .buy-link { display: block; background: white; border: 1px solid #3b82f6; color: #3b82f6; padding: 10px; border-radius: 8px; text-decoration: none; margin: 8px 0; text-align: center; font-weight: bold; }
            #loader { display: none; text-align: center; font-weight: bold; color: #ea580c; padding: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PartFinder Ultra</h1>
            <button class="btn btn-cam" onclick="document.getElementById('f').click()">📸 Scanner la pièce</button>
            <input type="file" id="f" accept="image/*" capture="environment" hidden onchange="pv(this)">
            <img id="preview">
            <textarea id="ctx" placeholder="Précisez le matériau ou l'usage (ex: Inox, 15/21...)"></textarea>
            <button id="go" class="btn btn-run" onclick="run()">LANCER L'IDENTIFICATION</button>
            <div id="loader">⚙️ Analyse multicouche...</div>
            <div id="res"></div>
        </div>
        <script>
            let img;
            function pv(i) { img = i.files[0]; const r = new FileReader(); r.onload=(e)=>{ const p=document.getElementById('preview'); p.src=e.target.result; p.style.display='block'; }; r.readAsDataURL(img); }
            async function run() {
                if(!img) return alert("Photo manquante");
                document.getElementById('loader').style.display="block";
                document.getElementById('go').style.display="none";
                const fd = new FormData(); fd.append('image', img); fd.append('context', document.getElementById('ctx').value);
                const r = await fetch('/identify', { method: 'POST', body: fd });
                document.getElementById('res').innerHTML = await r.text();
                document.getElementById('loader').style.display="none";
            }
        </script>
    </body>
    </html>
    """
