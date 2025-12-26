import os, base64, httpx, asyncio, json, re
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

# --- FILTRE DE SÉCURITÉ URL ---
def is_valid_product_link(url: str) -> bool:
    forbidden = ['/category', '/cat/', '/search', '/recherche', '/famille', '/resultats', 'filter=']
    return not any(word in url.lower() for word in forbidden)

# --- SOURCING OPTIMISÉ (SONAR API) ---
async def search_perplexity_async(query: str):
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key: return "⚠️ Clé API manquante."

    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # PROMPT OPTIMISÉ (Instructions Positives + XML Wrapper + JSON Schema)
    system_content = """
    <system_instructions>
    <role>Expert en Sourcing Industriel B2B/B2C</role>
    <task>Retourner UNIQUEMENT des liens directs vers fiches produits (SKU individuels).</task>
    <constraints>
      <constraint>Vérifier dimensions exactes (filetage, diamètre)</constraint>
      <constraint>Retourner uniquement URLs de produits précis</constraint>
    </constraints>
    <output_format>JSON array: [{"nom":"", "prix":"", "url":""}]</output_format>
    </system_instructions>
    """

    data = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Trouve produits pour : {query}"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "produits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "nom": {"type": "string"},
                                    "prix": {"type": "string"},
                                    "url": {"type": "string", "format": "uri"}
                                },
                                "required": ["nom", "url"]
                            }
                        }
                    }
                }
            }
        },
        "temperature": 0.1
    }

    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, json=data, headers=headers, timeout=28.0)
            if res.status_code != 200: return f"Erreur API ({res.status_code})"
            
            raw_data = res.json()['choices'][0]['message']['content']
            parsed = json.loads(raw_data)
            
            html_links = ""
            for item in parsed.get("produits", []):
                if is_valid_product_link(item['url']):
                    html_links += f'<a href="{item["url"]}" target="_blank" class="buy-link">🛒 {item["nom"]} - {item.get("prix", "Voir prix")}</a>'
            
            return html_links if html_links else "Aucune fiche produit directe trouvée."
        except Exception as e:
            return f"Délai dépassé ou erreur : {str(e)}"

# --- ANALYSE VISION ---
@app.post("/identify", response_class=HTMLResponse)
async def identify(image: UploadFile = File(...), context: str = Form("")):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    try:
        img_b64 = base64.b64encode(await image.read()).decode('utf-8')

        # Prompt court pour éviter l'overflow
        prompt = "ID TECHNIQUE. Format JSON: {\"mat\": \"\", \"std\": \"\", \"search\": \"\"}"

        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [{"type": "text", "text": f"{prompt} Context: {context}"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}],
            response_format={"type": "json_object"}
        )

        data = json.loads(completion.choices[0].message.content)
        links = await search_perplexity_async(data.get("search"))

        return f"""
        <div class="results animate-in">
            <div class="res-card mat"><strong>🧪 Matière</strong><p>{data.get('mat')}</p></div>
            <div class="res-card std"><strong>📏 Technique</strong><p>{data.get('std')}</p></div>
            <div class="res-card shop"><strong>🔗 Fiches Produits Directes</strong><div class="links-list">{links}</div></div>
            <button class="btn btn-run" onclick="location.reload()">🔄 Nouveau Diagnostic</button>
        </div>
        """
    except Exception as e:
        return f"<div class='res-card' style='color:red'>Erreur Vision : {str(e)}</div>"

@app.get("/", response_class=HTMLResponse)
def home():
    # Garde l'interface HTML identique (Micro + Caméra)
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
        <title>PartFinder PRO</title>
        <style>
            :root { --p: #ea580c; --b: #0f172a; }
            body { font-family: -apple-system, sans-serif; background: #f1f5f9; padding: 15px; margin: 0; }
            .container { max-width: 450px; margin: auto; background: white; border-radius: 20px; padding: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
            h1 { color: var(--p); text-align: center; margin-bottom: 25px; font-size: 1.5rem; }
            .btn { width: 100%; padding: 18px; border-radius: 12px; border: none; font-weight: bold; cursor: pointer; font-size: 1rem; margin-bottom: 12px; }
            .btn-cam { background: var(--p); color: white; }
            .btn-run { background: var(--b); color: white; }
            #preview { width: 100%; border-radius: 12px; margin-bottom: 15px; display: none; }
            .input-box { position: relative; margin-bottom: 15px; }
            textarea { width: 100%; padding: 15px; border: 1px solid #ddd; border-radius: 12px; box-sizing: border-box; resize: none; min-height: 70px; }
            .mic-btn { position: absolute; right: 10px; top: 12px; font-size: 1.5rem; background: none; border: none; cursor: pointer; }
            .res-card { padding: 15px; border-radius: 12px; margin-top: 15px; background: #fff; border: 1px solid #e2e8f0; }
            .mat { border-left: 5px solid var(--p); }
            .std { border-left: 5px solid #10b981; }
            .shop { border-left: 5px solid #3b82f6; background: #f0f9ff; }
            .buy-link { display: block; background: white; border: 1.5px solid #3b82f6; color: #3b82f6; padding: 12px; border-radius: 10px; text-decoration: none; font-weight: bold; margin: 10px 0; text-align: center; }
            #loader { display: none; text-align: center; color: var(--p); font-weight: bold; padding: 20px; }
            .animate-in { animation: fadeInUp 0.3s ease-out; }
            @keyframes fadeInUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PartFinder PRO</h1>
            <button class="btn btn-cam" onclick="document.getElementById('f').click()">📸 PRENDRE UNE PHOTO</button>
            <input type="file" id="f" accept="image/*" capture="environment" hidden onchange="pv(this)">
            <img id="preview">
            <div class="input-box">
                <textarea id="ctx" placeholder="Contexte technique..."></textarea>
                <button class="mic-btn" onclick="dictate()">🎙️</button>
            </div>
            <button id="go" class="btn btn-run" onclick="run()">LANCER L'IDENTIFICATION</button>
            <div id="loader">⚙️ Analyse & Sourcing Sonar...</div>
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
                } catch (e) { alert("Erreur connexion"); } 
                finally { document.getElementById('loader').style.display="none"; document.getElementById('go').style.display="block"; }
            }
        </script>
    </body>
    </html>
    """
