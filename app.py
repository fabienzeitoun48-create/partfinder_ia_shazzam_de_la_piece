import os, base64, httpx, asyncio, json, re
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

async def get_links(query):
    """Recherche Perplexity : Fiches produits uniquement."""
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"}
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Expert pièces détachées. Donne 3 liens de fiches produits réelles. Format: [Nom Article](URL) - Prix."},
            {"role": "user", "content": f"Trouver l'article exact : {query}"}
        ]
    }
    async with httpx.AsyncClient() as c:
        try:
            r = await c.post(url, json=payload, headers=headers, timeout=20)
            res = r.json()['choices'][0]['message']['content']
            # Conversion des liens markdown en boutons HTML
            return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank" class="buy-link">🛒 \1</a>', res).replace("\n", "<br>")
        except: return "Lien non trouvé."

@app.post("/identify")
async def identify(image: UploadFile = File(...), context: str = Form("")):
    try:
        img_b64 = base64.b64encode(await image.read()).decode('utf-8')
        
        # Prompt minimaliste pour éviter l'erreur 400 et la créativité inutile
        prompt = f"Identify this part. User info: {context}. Return ONLY json format: {{\"mat\": \"material\", \"std\": \"specs\", \"search\": \"precise search term\"}}"
        
        chat_completion = await asyncio.to_thread(client.chat.completions.create,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}],
            response_format={"type": "json_object"}
        )
        
        res_json = json.loads(chat_completion.choices[0].message.content)
        links_html = await get_links(res_json.get('search'))

        return f"""
        <div class="res-card"><strong>Composant :</strong> {res_json.get('search')}</div>
        <div class="res-card"><strong>Détails :</strong> {res_json.get('std')}</div>
        <div class="res-card" style="background:#f0f7ff; border-color:#3b82f6;"><strong>Résultats Shopping :</strong><br>{links_html}</div>
        """
    except Exception as e:
        return f"<div style='color:red; padding:10px; border:1px solid red;'>Erreur : {str(e)}</div>"

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
            body { font-family: sans-serif; background: #f4f7f6; display: flex; justify-content: center; padding: 20px; }
            .card { background: white; width: 100%; max-width: 400px; padding: 20px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
            .btn { width: 100%; padding: 15px; border: none; border-radius: 10px; font-weight: bold; cursor: pointer; margin-top: 10px; font-size: 1rem; }
            .btn-cam { background: #ea580c; color: white; }
            .btn-run { background: #1e293b; color: white; }
            #preview { width: 100%; border-radius: 10px; margin-top: 15px; display: none; }
            .input-group { position: relative; margin-top: 15px; }
            textarea { width: 100%; height: 80px; padding: 10px; border-radius: 10px; border: 1px solid #ddd; box-sizing: border-box; resize: none; }
            .mic-btn { position: absolute; right: 10px; top: 10px; background: none; border: none; font-size: 20px; cursor: pointer; }
            .res-card { padding: 12px; border-radius: 8px; border: 1px solid #eee; margin-top: 10px; background: #fff; }
            .buy-link { display: block; padding: 10px; margin: 8px 0; border: 1px solid #3b82f6; border-radius: 5px; color: #3b82f6; text-decoration: none; text-align: center; font-weight: bold; font-size: 0.9rem; }
            #loading { display: none; text-align: center; color: #ea580c; font-weight: bold; padding: 10px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2 style="color:#ea580c; text-align:center;">PartFinder</h2>
            <button class="btn btn-cam" onclick="document.getElementById('f').click()">📷 PRENDRE UNE PHOTO</button>
            <input type="file" id="f" accept="image/*" capture="environment" hidden onchange="p(this)">
            <img id="preview">
            
            <div class="input-group">
                <textarea id="t" placeholder="Infos complémentaires (optionnel)..."></textarea>
                <button class="mic-btn" onclick="s()">🎙️</button>
            </div>
            
            <button class="btn btn-run" id="rb" onclick="u()">LANCER L'IDENTIFICATION</button>
            <div id="loading">Recherche en cours...</div>
            <div id="res"></div>
        </div>

        <script>
            let img;
            function p(i) { img=i.files[0]; const r=new FileReader(); r.onload=(e)=>{ const v=document.getElementById('preview'); v.src=e.target.result; v.style.display='block'; }; r.readAsDataURL(img); }
            
            function s() {
                const S = window.SpeechRecognition || window.webkitSpeechRecognition;
                if(!S) return alert("Micro non supporté");
                const r = new S(); r.lang='fr-FR'; r.onresult=(e)=>{ document.getElementById('t').value=e.results[0][0].transcript; }; r.start();
            }

            async function u() {
                if(!img) return alert("Photo manquante");
                document.getElementById('loading').style.display='block';
                document.getElementById('rb').style.display='none';
                const fd = new FormData(); fd.append('image', img); fd.append('context', document.getElementById('t').value);
                try {
                    const r = await fetch('/identify', { method: 'POST', body: fd });
                    document.getElementById('res').innerHTML = await r.text();
                } catch(e) { alert("Erreur serveur"); }
                finally { document.getElementById('loading').style.display='none'; document.getElementById('rb').style.display='block'; }
            }
        </script>
    </body>
    </html>
    """
