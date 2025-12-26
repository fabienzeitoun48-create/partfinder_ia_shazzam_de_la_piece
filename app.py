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
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --- FILTRE DE SÉCURITÉ URL ---
def is_valid_product_link(url: str) -> bool:
    """Vérifie si le lien pointe vers une fiche produit unitaire (SKU)."""
    forbidden = ['/category', '/cat/', '/search', '/recherche', '/famille', '/resultats', 'filter=', '/shop?', 'page=']
    return not any(word in url.lower() for word in forbidden) and url.startswith('http')

# --- SOURCING OPTIMISÉ (PERPLEXITY SONAR) ---
async def search_perplexity_async(query: str):
    """
    Recherche optimisée Sonar avec système prompt réduit et params API.
    Retourne JSON structuré avec liens filtrés.
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        return json.dumps({"error": "Clé API Perplexity manquante.", "produits": []})
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # ✅ PROMPT OPTIMISÉ : ~120 mots, instructions positives, JSON structuré
    system_content = (
        "Tu es expert en sourcing industriel B2B/B2C. "
        "Tâche: Retourner UNIQUEMENT liens directs vers fiches produits (SKU individuels). "
        "Règles: URLs valides vers produits précis. Vérifier dimensions exactes. "
        "Si dimensions manquent: poser 1 question précise. "
        "Retourner JSON array uniquement: [{\"nom\",\"marque\",\"modele\",\"taille\",\"prix\",\"url\"}]"
    )

    data = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Trouve fiches produits pour: {query}"}
        ],
        "temperature": 0,
        "max_tokens": 2000,
        # ✅ PARAMS API (pas dans le prompt !)
        # "search_domain_filter": ["rs-online.com", "wesco.fr", "dkc.de", "amazon.fr", "cdiscount.com"],
        # "search_context_size": "high"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(url, json=data, headers=headers, timeout=28.0)
            
            if res.status_code != 200:
                return json.dumps({
                    "error": f"Erreur API Perplexity ({res.status_code})",
                    "produits": []
                })
            
            content = res.json()['choices'][0]['message']['content']
            
            # ✅ Parse JSON réponse ou fallback sur extraction regex
            try:
                produits = json.loads(content)
                if not isinstance(produits, list):
                    produits = [produits]
            except json.JSONDecodeError:
                # Fallback: extraire URLs + formatter
                produits = []
                url_pattern = r'(https?://[^\s\)]+)'
                urls = re.findall(url_pattern, content)
                for u in urls:
                    if is_valid_product_link(u):
                        produits.append({
                            "nom": "Produit détecté",
                            "marque": "Non spécifiée",
                            "modele": "Non spécifiée",
                            "taille": "Non spécifiée",
                            "prix": "Voir lien",
                            "url": u
                        })
            
            # ✅ Filtrer les liens invalides
            produits = [p for p in produits if is_valid_product_link(p.get("url", ""))]
            
            return json.dumps({"produits": produits, "raw_content": content})
        
        except asyncio.TimeoutError:
            return json.dumps({
                "error": "Délai dépassé (serveur Perplexity occupé)",
                "produits": []
            })
        except Exception as e:
            return json.dumps({
                "error": f"Erreur recherche: {str(e)}",
                "produits": []
            })

# --- ANALYSE VISION (LLAMA 4 SCOUT) ---
@app.post("/identify", response_class=HTMLResponse)
async def identify(image: UploadFile = File(...), context: str = Form("")):
    """
    Identifie pièce via vision + lance sourcing.
    Retourne HTML avec résultats structurés.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "<div class='res-card' style='color:red;'>⚠️ Clé API Groq manquante</div>"
    
    client = Groq(api_key=api_key)
    
    try:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # ✅ PROMPT VISION OPTIMISÉ
        prompt = """
IDENTIFICATION TECHNIQUE RAPIDE.
Réponds STRICTEMENT en JSON:
{
  "mat": "Matière exact + état (ex: Laiton nickelé, usure légère)",
  "std": "Dimensions précises + type (ex: M6x30, Vis inox)",
  "search": "Terme de recherche ultra-spécifique pour sourcing direct"
}
NE RIEN AJOUTER."""
        
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}\n\nContexte supplémentaire: {context if context else 'Aucun'}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    }
                ]
            }],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=500
        )
        
        # ✅ Parse réponse JSON vision
        vision_data = json.loads(completion.choices[0].message.content)
        search_query = vision_data.get("search", "produit")
        
        # ✅ Lancer sourcing avec requête précise
        sourcing_result = await search_perplexity_async(search_query)
        sourcing_data = json.loads(sourcing_result)
        
        # ✅ Construire HTML réponse avec produits trouvés
        produits_html = ""
        if sourcing_data.get("produits"):
            for prod in sourcing_data["produits"]:
                url = prod.get("url", "#")
                nom = prod.get("nom", "Produit")
                marque = prod.get("marque", "")
                modele = prod.get("modele", "")
                taille = prod.get("taille", "")
                prix = prod.get("prix", "Voir lien")
                
                # Format: Marque + Modèle + Taille - Prix
                titre = f"{marque} {modele} {taille}".strip() or nom
                
                produits_html += f"""
                <a href="{url}" target="_blank" class="buy-link">
                    <strong>{titre}</strong>
                    <span class="prix">{prix}</span>
                </a>
                """
        else:
            produits_html = f"""
            <div style="color:#666; font-size:0.9rem; padding:10px; text-align:center;">
                ℹ️ {sourcing_data.get('error', 'Aucun produit trouvé directement')}
                <br><small>Essayez une recherche manuelle ou ajustez le contexte</small>
            </div>
            """
        
        return f"""
        <div class="results animate-in">
            <div class="res-card mat">
                <strong>🧪 Matière & État</strong>
                <p>{vision_data.get('mat', 'Analyse...')}</p>
            </div>
            <div class="res-card std">
                <strong>📏 Dimensions & Type</strong>
                <p>{vision_data.get('std', 'Analyse...')}</p>
            </div>
            <div class="res-card shop">
                <strong>🔗 Fiches Produits Directes</strong>
                <div class="links-list">{produits_html}</div>
            </div>
            <button class="btn btn-run" onclick="location.reload()">🔄 Nouveau Diagnostic</button>
        </div>
        """
    
    except json.JSONDecodeError as e:
        return f"""
        <div class='res-card' style='color:#d97706; border-left: 5px solid #d97706;'>
            ⚠️ Erreur parsing JSON: {str(e)[:100]}
            <br><small>La pièce est peut-être incompréhensible. Réessayez avec un meilleur angle.</small>
        </div>
        """
    except Exception as e:
        return f"""
        <div class='res-card' style='color:red; border-left: 5px solid red;'>
            ❌ Erreur: {str(e)[:150]}
        </div>
        """

# --- INTERFACE WEB ---
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
        <title>PartFinder PRO - Sourcing Intelligent</title>
        <style>
            :root { 
                --primary: #ea580c; 
                --dark: #0f172a;
                --success: #10b981;
                --info: #3b82f6;
                --warning: #d97706;
            }
            
            * { box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #f1f5f9 0%, #e0e7ff 100%);
                padding: 15px;
                margin: 0;
                min-height: 100vh;
            }
            
            .container { 
                max-width: 480px; 
                margin: 20px auto;
                background: white; 
                border-radius: 20px; 
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.12);
            }
            
            h1 { 
                color: var(--primary); 
                text-align: center; 
                margin: 0 0 10px 0;
                font-size: 1.8rem;
                font-weight: 700;
            }
            
            .subtitle {
                text-align: center;
                color: #666;
                font-size: 0.9rem;
                margin-bottom: 25px;
            }
            
            .btn { 
                width: 100%; 
                padding: 16px;
                border-radius: 12px; 
                border: none; 
                font-weight: 600;
                cursor: pointer; 
                font-size: 1rem;
                margin-bottom: 12px;
                transition: all 0.2s ease;
            }
            
            .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
            .btn:active { transform: translateY(0); }
            
            .btn-cam { 
                background: var(--primary); 
                color: white;
            }
            
            .btn-cam:hover { background: #d94c0a; }
            
            .btn-run { 
                background: var(--dark); 
                color: white;
            }
            
            .btn-run:hover { background: #1e293b; }
            
            .btn-run:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            #preview { 
                width: 100%; 
                border-radius: 12px; 
                margin-bottom: 15px; 
                display: none; 
                border: 2px solid #e2e8f0;
                max-height: 300px;
                object-fit: cover;
            }
            
            .input-box { 
                position: relative; 
                margin-bottom: 15px;
            }
            
            textarea { 
                width: 100%; 
                padding: 12px;
                border: 1.5px solid #e2e8f0;
                border-radius: 10px;
                box-sizing: border-box; 
                font-family: inherit;
                font-size: 0.95rem;
                resize: none; 
                min-height: 60px;
                transition: border-color 0.2s;
            }
            
            textarea:focus {
                outline: none;
                border-color: var(--primary);
            }
            
            .mic-btn { 
                position: absolute; 
                right: 10px; 
                top: 50%;
                transform: translateY(-50%);
                font-size: 1.3rem; 
                background: none; 
                border: none; 
                cursor: pointer;
                padding: 5px;
            }
            
            .mic-btn:hover { opacity: 0.7; }
            
            /* Cartes résultats */
            .res-card { 
                padding: 15px; 
                border-radius: 12px; 
                margin-top: 15px; 
                background: white;
                border-left: 5px solid;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            }
            
            .res-card strong {
                display: block;
                margin-bottom: 8px;
                font-size: 1rem;
            }
            
            .res-card p {
                margin: 0;
                line-height: 1.5;
                color: #333;
            }
            
            .mat { border-left-color: var(--primary); }
            .std { border-left-color: var(--success); }
            .shop { 
                border-left-color: var(--info);
                background: #f0f9ff;
            }
            
            /* Liens produits */
            .links-list { margin-top: 12px; }
            
            .buy-link { 
                display: block; 
                background: white;
                border: 1.5px solid var(--info);
                color: var(--info);
                padding: 12px;
                border-radius: 10px;
                text-decoration: none;
                font-weight: 600;
                margin: 10px 0;
                text-align: center;
                transition: all 0.2s;
                position: relative;
            }
            
            .buy-link:hover {
                background: var(--info);
                color: white;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
            }
            
            .buy-link strong {
                display: block;
                margin-bottom: 4px;
                font-size: 0.95rem;
            }
            
            .buy-link .prix {
                font-size: 0.85rem;
                opacity: 0.8;
            }
            
            /* Loader */
            #loader { 
                display: none; 
                text-align: center; 
                color: var(--primary);
                font-weight: 600;
                padding: 30px 20px;
                animation: pulse 1.5s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.6; }
            }
            
            .spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid var(--primary);
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            /* Animations */
            .animate-in { 
                animation: fadeInUp 0.4s ease-out;
            }
            
            @keyframes fadeInUp { 
                from { 
                    opacity: 0; 
                    transform: translateY(15px); 
                } 
                to { 
                    opacity: 1; 
                    transform: translateY(0); 
                } 
            }
            
            /* Responsive */
            @media (max-width: 480px) {
                .container {
                    padding: 20px;
                    margin: 10px;
                }
                h1 {
                    font-size: 1.5rem;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📷 PartFinder PRO</h1>
            <p class="subtitle">Identification + Sourcing Intelligent</p>
            
            <button class="btn btn-cam" onclick="document.getElementById('f').click()">
                📸 PHOTOGRAPHIER LA PIÈCE
            </button>
            <input type="file" id="f" accept="image/*" capture="environment" hidden onchange="pv(this)">
            <img id="preview">
            
            <div class="input-box">
                <textarea 
                    id="ctx" 
                    placeholder="Contexte (ex: robinet fuyant, diamètre extérieur 15mm)..."
                ></textarea>
                <button class="mic-btn" onclick="dictate()" title="Dictée vocale">🎙️</button>
            </div>
            
            <button id="go" class="btn btn-run" onclick="run()">
                🚀 LANCER L'IDENTIFICATION
            </button>
            
            <div id="loader">
                <span class="spinner"></span>
                ⚙️ Analyse Vision Llama Scout & Sourcing...
            </div>
            
            <div id="res"></div>
        </div>

        <script>
            let img = null;
            
            // ✅ Restaurer résultats précédents
            window.onload = () => {
                const saved = localStorage.getItem('partfinder_last_scan');
                if(saved) {
                    document.getElementById('res').innerHTML = saved;
                }
            };

            // ✅ Prévisualiser image
            function pv(input) {
                img = input.files[0];
                if(!img) return;
                
                const reader = new FileReader();
                reader.onload = (e) => { 
                    const preview = document.getElementById('preview');
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(img);
            }

            // ✅ Dictée vocale (français)
            function dictate() {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                if(!SpeechRecognition) {
                    alert("Dictée vocale non supportée par votre navigateur");
                    return;
                }
                
                const sr = new SpeechRecognition();
                sr.lang = 'fr-FR';
                sr.continuous = false;
                sr.interimResults = false;
                
                sr.onstart = () => {
                    document.querySelector('.mic-btn').style.opacity = '0.5';
                };
                
                sr.onresult = (event) => {
                    if(event.results.length > 0) {
                        const transcript = event.results[0][0].transcript;
                        document.getElementById('ctx').value = transcript;
                    }
                };
                
                sr.onerror = (event) => {
                    console.error('Erreur dictée:', event.error);
                };
                
                sr.onend = () => {
                    document.querySelector('.mic-btn').style.opacity = '1';
                };
                
                sr.start();
            }

            // ✅ Lancer analyse + sourcing
            async function run() {
                if(!img) {
                    alert("📸 Veuillez d'abord prendre une photo");
                    return;
                }
                
                document.getElementById('loader').style.display = "block";
                document.getElementById('go').disabled = true;
                document.getElementById('res').innerHTML = "";
                
                const formData = new FormData();
                formData.append('image', img);
                formData.append('context', document.getElementById('ctx').value || "");
                
                try {
                    const response = await fetch('/identify', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if(!response.ok) {
                        throw new Error(`Erreur HTTP ${response.status}`);
                    }
                    
                    const html = await response.text();
                    document.getElementById('res').innerHTML = html;
                    
                    // ✅ Sauvegarder résultat
                    localStorage.setItem('partfinder_last_scan', html);
                    
                } catch (error) {
                    document.getElementById('res').innerHTML = \`
                        <div class='res-card' style='border-left-color: #d97706; background: #fef3c7;'>
                            <strong>❌ Erreur de connexion</strong>
                            <p style='color: #92400e; margin-top: 8px;'>\${error.message}</p>
                            <small>Vérifiez votre connexion internet et réessayez.</small>
                        </div>
                    \`;
                    console.error(error);
                    
                } finally {
                    document.getElementById('loader').style.display = "none";
                    document.getElementById('go').disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
