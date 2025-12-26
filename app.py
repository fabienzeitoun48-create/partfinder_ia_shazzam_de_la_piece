import os, base64, httpx, asyncio, json, re
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialisation de tous les moteurs
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

async def call_perplexity(query):
    """Sourcing profond avec Perplexity."""
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"}
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Tu es un expert en pièces détachées industrielles et plomberie. Trouve 3 fiches produits précises (Leroy Merlin, ManoMano, Cedeo, RS). Pas de catégories. Format: [Nom de la pièce](URL) - Prix."},
            {"role": "user", "content": f"Trouve la fiche technique et prix pour : {query}"}
        ]
    }
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(url, json=payload, headers=headers, timeout=25.0)
            content = r.json()['choices'][0]['message']['content']
            # Nettoyage des liens
            html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank" class="buy-link">🛒 \1</a>', content)
            return html.replace("\n", "<br>")
        except:
            return "⚠️ Recherche impossible. Vérifiez votre clé Perplexity."

async def deepseek_refiner(llama_info, mistral_info, user_context):
    """Le cerveau DeepSeek qui croise les données pour créer la requête parfaite."""
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"}
    prompt = f"""
    Analyse de deux modèles vision :
    1. {llama_info}
    2. {mistral_info}
    Contexte client : {user_context}

    Ta mission : Créer une requête de recherche ultra-technique (marque, cotes, matière).
    Réponds impérativement en format JSON : 
    {{"search_query": "ton terme de recherche", "specs": "résumé technique"}}
    """
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(url, headers=headers, json={
                "model": "deepseek-chat", 
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }, timeout=15)
            return r.json()['choices'][0]['message']['content']
        except:
            return json.dumps({"search_query": user_context, "specs": "Analyse simplifiée"})

@app.post("/identify")
async def identify(image: UploadFile = File(...), context: str = Form("")):
    try:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        # 1. Analyse Vision simultanée (Groq + Mistral)
        # Groq (Llama 4 Scout) - Fix erreur 400 JSON
        l_task = asyncio.to_thread(groq_client.chat.completions.create,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Identify part. Return json: {mat, std, search}"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}],
            response_format={"type": "json_object"}
        )

        m_task = mistral_client.chat.complete_async(
            model="pixtral-12b-2409",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Décris techniquement cette pièce."}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}]}]
        )

        l_res, m_res = await asyncio.gather(l_task, m_task)
        
        # 2. Raffinement par DeepSeek
        ds_raw = await deepseek_refiner(l_res.choices[0].message.content, m_res.choices[0].message.content, context)
        ds_data = json.loads(ds_raw)

        # 3. Sourcing final Perplexity
        links = await call_perplexity(ds_data['search_query'])

        return f"""
        <div class="res-card mat"><strong>🛠️ Identification :</strong><br>{ds_data['specs']}</div>
        <div class="res-card shop"><strong>🛒 Offres Trouvées :</strong><br>
            <p style="font-size:0.7rem; color:gray;">Terme utilisé : {ds_data['search_query']}</p>
            {links}
        </div>
        """
    except Exception as e:
        return f"<div style='color:red'>Erreur Système : {str(e)}</div>"

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
            body { font-family: -apple-system, sans-serif; background: #f1f5f9; padding: 15px; margin: 0; }
            .container { max-width: 450px; margin: auto; background: white; border-radius: 20px; padding: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
            .btn { width: 100%; padding: 18px; border-radius: 12px; border: none; font-weight: bold; cursor: pointer; font-size: 1rem; margin-top: 10px; }
            .btn-cam { background: #ea580c; color: white; }
            .btn-run { background: #0f172a; color: white; }
            #preview { width: 100%; border-radius: 12px; margin-top: 15px; display: none; }
            .input-box { position: relative; margin-top: 15px; }
            textarea { width: 100%; padding: 15px; border: 1px solid #ddd; border-radius: 12px; box-sizing: border-box; min-height: 80px; font-family: inherit; }
            .mic-btn { position: absolute; right: 10px; top: 12px; font-size: 1.5rem; background: none; border: none; cursor: pointer; }
            .res-card { padding: 15px; border-radius: 12px; margin-top: 15px; border: 1px solid #e2e8f0; line-height: 1.5; }
            .mat { border-left: 5px solid #ea580c; background: #fffaf8; }
            .shop { border-left: 5px solid #3b82f6; background: #f0f9ff; }
            .buy-link { display: block; background: white; border: 1.5px solid #3b82f6; color: #3b82f6; padding: 12px; border-radius: 10px; text-decoration: none; font-weight: bold; margin: 10px 0; text-align: center; }
            #loader { display: none; text-align: center; color: #ea580c; font-weight: bold; padding: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 style="text-align:center; color:#ea580c;">PartFinder Pro</h2>
            <button class="btn btn-cam" onclick="document.getElementById('f').click()">📸 PHOTO DE LA PIÈCE</button>
            <input type="file" id="f" accept="image/*" capture="environment" hidden onchange="pv(this)">
            <img id="preview">
            <div class="input-box">
                <textarea id="ctx" placeholder="Précisez le contexte..."></textarea>
                <button class="mic-btn" onclick="dictate()">🎙️</button>
            </div>
            <button id="go" class="btn btn-run" onclick="run()">LANCER LE DIAGNOSTIC</button>
            <div id="loader">⚙️ Triple Analyse (Llama + Mistral + DeepSeek)...</div>
            <div id="res"></div>
        </div>
        <script>
            let img;
            function pv(i) { img = i.files[0]; const r = new FileReader(); r.onload = (e) => { const p = document.getElementById('preview'); p.src = e.target.result; p.style.display = 'block'; }; r.readAsDataURL(img); }
            function dictate() {
                const sr = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                sr.lang = 'fr-FR';
                sr.onresult = (e) => { document.getElementById('ctx').value = e.results[0][0].transcript; };
                sr.start();
            }
            async function run() {
                if(!img) return alert("Photo requise");
                document.getElementById('loader').style.display="block";
                document.getElementById('go').style.display="none";
                const fd = new FormData(); fd.append('image', img); fd.append('context', document.getElementById('ctx').value);
                try {
                    const r = await fetch('/identify', { method: 'POST', body: fd });
                    document.getElementById('res').innerHTML = await r.text();
                } catch (e) { alert("Erreur serveur"); }
                finally { document.getElementById('loader').style.display="none"; document.getElementById('go').style.display="block"; }
            }
        </script>
    </body>
    </html>
    """
