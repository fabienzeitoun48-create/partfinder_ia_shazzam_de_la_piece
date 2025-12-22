import os
import re
import base64
import httpx
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv
import database_standards as db

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

NOM_PROJET = "PartFinder AI"

async def search_parts_web(technical_desc: str):
    """Cherche la pièce sur les sites marchands via Perplexity"""
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key: return "Recherche marchande indisponible."
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "Tu es un expert en sourcing de pièces détachées (Plomberie/Quincaillerie). Trouve des liens d'achat (ManoMano, Leroy Merlin, Cédéo) pour la pièce décrite."},
            {"role": "user", "content": f"Trouve cette pièce ou son équivalent : {technical_desc}"}
        ]
    }
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, json=data, headers=headers, timeout=25.0)
            return res.json()['choices'][0]['message']['content']
        except: return "Erreur lors de la recherche de stocks."

def format_output(text: str, web_info: str) -> str:
    # Transformation simple du Markdown en HTML structuré
    sections = text.split("##")
    html = ""
    for s in sections:
        if not s.strip(): continue
        parts = s.split("\n", 1)
        title = parts[0].strip()
        body = parts[1].replace("\n", "<br>") if len(parts) > 1 else ""
        html += f"<div class='section'><h3>{title}</h3><p>{body}</p></div>"
    
    if web_info:
        web_body = web_info.replace("\n", "<br>")
        html += f"<div class='section web'><h3>🛒 DISPONIBILITÉS ET ACHAT</h3><p>{web_body}</p></div>"
    return html

@app.post("/identify")
async def identify_part(image: UploadFile = File(...), context: str = Form("")):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    img_b64 = base64.b64encode(await image.read()).decode('utf-8')
    
    prompt = f"""Tu es un expert mondial en pièces détachées de bâtiment. 
    Analyse cette photo et utilise ces standards techniques : {db.get_standards_summary()}.
    
    Réponds suivant ce format :
    ## 🔧 Identification Technique
    (Nom exact de la pièce, matériau, état d'usure)
    
    ## 📏 Dimensions Estimées
    (Filetage probable ex: 15/21, diamètre, longueur)
    
    ## 💡 Conseil de Remplacement
    (Quelle pièce moderne choisir ? Faut-il tout changer ou juste un joint ?)
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": f"Contexte additionnel: {context}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]}],
            temperature=0.1
        )
        analysis = completion.choices[0].message.content
        web_info = await search_parts_web(analysis[:200])
        return HTMLResponse(content=format_output(analysis, web_info))
    except Exception as e:
        return HTMLResponse(content=f"Erreur d'analyse : {str(e)}")

@app.get("/")
def home():
    return f"""
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: sans-serif; background: #f3f4f6; padding: 20px; color: #374151; }}
            .container {{ max-width: 500px; margin: auto; background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            h1 {{ color: #ea580c; text-align: center; font-size: 1.5rem; }}
            .section {{ border-left: 4px solid #ea580c; background: #fff7ed; padding: 15px; margin-bottom: 15px; border-radius: 0 8px 8px 0; }}
            .web {{ border-left-color: #2563eb; background: #eff6ff; }}
            #preview {{ width: 100%; border-radius: 10px; margin-bottom: 15px; display: none; }}
            .btn {{ background: #ea580c; color: white; border: none; width: 100%; padding: 15px; border-radius: 8px; font-weight: bold; cursor: pointer; }}
            #loading {{ display: none; text-align: center; font-weight: bold; color: #ea580c; margin: 15px 0; }}
            input[type="file"] {{ display: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 PartFinder AI</h1>
            <p style="text-align:center; font-size: 0.9rem;">Le Shazam des pièces détachées</p>
            <img id="preview">
            <button class="btn" onclick="document.getElementById('file').click()" style="background:#6b7280; margin-bottom:10px;">📸 PHOTOGRAPHIER LA PIÈCE</button>
            <input type="file" id="file" accept="image/*" capture="environment" onchange="showPv(this)">
            <textarea id="ctx" style="width:100%; height:60px; border-radius:8px; border:1px solid #d1d5db; padding:10px;" placeholder="Ex: C'est pour un évier de cuisine..."></textarea>
            <button class="btn" onclick="analyze()" style="margin-top:10px;">IDENTIFIER ET TROUVER</button>
            <div id="loading">Identification en cours...</div>
            <div id="result"></div>
        </div>
        <script>
            function showPv(i) {{
                const r = new FileReader();
                r.onload = (e) => {{ const p = document.getElementById('preview'); p.src = e.target.result; p.style.display='block'; }};
                r.readAsDataURL(i.files[0]);
            }}
            async function analyze() {{
                const res = document.getElementById('result');
                const load = document.getElementById('loading');
                res.innerHTML = ""; load.style.display = "block";
                const fd = new FormData();
                fd.append('image', document.getElementById('file').files[0]);
                fd.append('context', document.getElementById('ctx').value);
                const r = await fetch('/identify', {{ method: 'POST', body: fd }});
                res.innerHTML = await r.text();
                load.style.display = "none";
            }}
        </script>
    </body>
    </html>
    """