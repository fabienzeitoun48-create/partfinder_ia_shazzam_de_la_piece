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

# --- AGENT DE RAISONNEMENT : DEEPSEEK (Fix Error 400 JSON) ---
async def deepseek_consensus(llama_txt, mistral_txt, user_ctx):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    url = "https://api.deepseek.com/v1/chat/completions"
    
    # Le mot "JSON" est maintenant explicitement présent pour éviter l'erreur 400
    prompt = f"""
    CONTEXTE UTILISATEUR : {user_ctx}
    RAPPORT 1 : {llama_txt}
    RAPPORT 2 : {mistral_txt}
    
    Tu es un expert technique. Génère un terme de recherche ultra-précis pour trouver cet objet exact.
    Si le contexte mentionne INOX, le terme doit inclure 'Acier Inox'.
    
    Réponds obligatoirement sous la forme d'un objet JSON :
    {{
      "matiere": "verdict sur le matériau",
      "technique": "dimensions ou normes détectées",
      "search": "requête précise pour Google Shopping"
    }}
    """
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, 
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "deepseek-chat", 
                    "messages": [{"role": "user", "content": prompt}], 
                    "response_format": {"type": "json_object"} # Exige le mot "JSON" dans le prompt
                },
                timeout=15.0)
            return json.loads(res.json()['choices'][0]['message']['content'])
        except Exception as e:
            return {"matiere": "Analyse standard", "technique": "N/A", "search": user_ctx or "pièce détachée"}

# --- SOURCING : PERPLEXITY ---
async def search_perplexity_pro(query: str):
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    url = "https://api.perplexity.ai/chat/completions"
    
    data = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Trouve 3 liens directs de fiches produits (pas de catégories). Inclus le nom exact et le prix. Format : [Nom](URL)."},
            {"role": "user", "content": f"Fiche produit exacte pour : {query}"}
        ]
    }
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=data, timeout=25.0)
            content = res.json()['choices'][0]['message']['content']
            html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank" class="buy-link">🛒 \1</a>', content)
            return html.replace("\n", "<br>")
        except:
            return "Aucun lien trouvé."

@app.post("/identify", response_class=HTMLResponse)
async def identify(image: UploadFile = File(...), context: str = Form("")):
    try:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        # Vision Llama (Correction JSON ici aussi)
        l_task = asyncio.to_thread(groq_client.chat.completions.create,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Identifie l'objet en format JSON : mat, std, search."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}],
            response_format={"type": "json_object"}
        )

        m_res = await mistral_client.chat.complete_async(
            model="pixtral-12b-2409",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Analyse technique précise de cet objet."}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}]}]
        )

        l_res = await l_task
        final_data = await deepseek_consensus(l_res.choices[0].message.content, m_res.choices[0].message.content, context)
        links_html = await search_perplexity_pro(final_data['search'])

        return f"""
        <div class="results animate-in">
            <div class="res-card mat"><strong>🧪 Matériau</strong><p>{final_data['matiere']}</p></div>
            <div class="res-card std"><strong>📐 Fiche technique</strong><p>{final_data['technique']}</p></div>
            <div class="res-card shop"><strong>🔗 Liens directs</strong>
                <p style="font-size:0.7rem; color:#666">Cible : {final_data['search']}</p>
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
        <title>PartFinder Pro</title>
        <style>
            body { font-family: sans-serif; background: #f0f4f8; padding: 15px; margin: 0; }
            .container { max-width: 450px; margin: auto; background: white; border-radius: 20px; padding: 20px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
            .btn { width: 100%; padding: 16px; border-radius: 12px; border: none; font-weight: bold; cursor: pointer; font-size: 1rem; }
            .btn-cam { background: #ea580c; color: white; margin-bottom: 10px; }
            .btn-run { background: #1e293b; color: white; margin-top: 10px; }
            #preview { width: 100%; border-radius: 12px; display: none; margin-top: 10px; }
            .input-box { position: relative; margin-top: 15px; }
            textarea { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 10px; box-sizing: border-box; min-height: 80px; }
            .mic-btn { position: absolute; right: 10px; top: 10px; font-size: 1.5rem; background: none; border: none; cursor: pointer; }
            .res-card { padding: 15px; border-radius: 10px; margin-top: 15px; border: 1px solid #eee; }
            .mat { border-left: 5px solid #ea580c; }
            .std { border-left: 5px solid #10b981; }
            .shop { border-left: 5px solid #3b82f6; background: #f0f7ff; }
            .buy-link { display: block; background: white; border: 1px solid #3b82f6; color: #3b82f6; padding: 10px; border-radius: 8px; text-decoration: none; margin: 8px 0; text-align: center; font-weight: bold; }
            #loader { display: none; text-align: center; padding: 20px; color: #ea580c; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 style="text-align:center; color:#ea580c;">PartFinder Pro</h2>
            <button class="btn btn-cam" onclick="document.getElementById('f').click()">📸 Photo de la pièce</button>
            <input type="file" id="f" accept="image/*" capture="environment" hidden onchange="pv(this)">
            <img id="preview">
            
            <div class="input-box">
                <textarea id="ctx" placeholder="Précisez le matériau (Inox, Laiton...)"></textarea>
                <button class="mic-btn" onclick="dictate()">🎙️</button>
            </div>
            
            <button id="go" class="btn btn-run" onclick="run()">Lancer l'Analyse</button>
            <div id="loader">⚙️ Triple Analyse + Sourcing...</div>
            <div id="res"></div>
        </div>
        <script>
            let img;
            function pv(i) { img = i.files[0]; const r = new FileReader(); r.onload=(e)=>{ const p=document.getElementById('preview'); p.src=e.target.result; p.style.display='block'; }; r.readAsDataURL(img); }
            
            function dictate() {
                const sr = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                sr.lang = 'fr-FR';
                sr.onresult = (e) => { document.getElementById('ctx').value = e.results[0][0].transcript; };
                sr.start();
            }

            async function run() {
                if(!img) return alert("Photo manquante");
                document.getElementById('loader').style.display="block";
                document.getElementById('go').style.display="none";
                const fd = new FormData(); fd.append('image', img); fd.append('context', document.getElementById('ctx').value);
                try {
                    const r = await fetch('/identify', { method: 'POST', body: fd });
                    document.getElementById('res').innerHTML = await r.text();
                } catch (e) { alert("Erreur réseau"); } 
                finally { document.getElementById('loader').style.display="none"; }
            }
        </script>
    </body>
    </html>
    """
