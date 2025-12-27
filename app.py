import os
import base64
import httpx
import asyncio
import json
import re
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv
from PIL import Image, ImageStat
import numpy as np
from functools import wraps

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# -------------------------
# Configuration production
# -------------------------
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_TIMEOUT = 28.0
PERPLEXITY_RETRIES = 2
PERPLEXITY_BACKOFF = 1.2

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_TIMEOUT = 30.0

# Domains considered trustworthy for product pages (extend as needed)
WHITELIST_DOMAINS = {
    "amazon.fr", "amazon.com", "manomano.fr", "leroymerlin.fr",
    "ebay.fr", "aliexpress.com", "rs-online.com", "mouser.com",
    "conrad.com", "farnell.com", "rs-components.com", "digikey.fr", "digikey.com"
}
BLACKLIST_PATTERNS = ['/category', '/cat/', '/search', '/recherche', '/famille', '/resultats', 'filter=', '/collections/']

# Simple in-memory TTL cache for URL validation to reduce load
_URL_VALIDATION_CACHE: Dict[str, Dict[str, Any]] = {}
_URL_CACHE_TTL = 60 * 60 * 24  # 24h

# -------------------------
# Utilities
# -------------------------
def ttl_cache(ttl_seconds: int):
    def decorator(func):
        cache = {}

        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = (func.__name__, args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in cache:
                value, ts = cache[key]
                if now - ts < ttl_seconds:
                    return value
            value = await func(*args, **kwargs)
            cache[key] = (value, now)
            return value
        return wrapper
    return decorator

def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace('www.', '')
    except Exception:
        return ""

def is_valid_product_link(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    lower = url.lower()
    if any(p in lower for p in BLACKLIST_PATTERNS):
        return False
    # basic scheme check
    if not lower.startswith("http"):
        return False
    return True

def looks_like_product_page_text(html_text: str) -> bool:
    checks = [
        r'og:price:amount',
        r'itemprop=["\']price',
        r'itemprop=["\']sku',
        r'\"price\" *: *\"?\d',
        r'\b(réf|référence|sku|part ?no|partnumber)\b',
        r'\b\d{1,4}\s?mm\b',
    ]
    for pattern in checks:
        if re.search(pattern, html_text, flags=re.IGNORECASE):
            return True
    return False

# -------------------------
# Image quality checks
# -------------------------
def pil_image_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io_bytes := (lambda x: Image.open(io := __import__("io").BytesIO(x)))(b))

def image_brightness(img: Image.Image) -> float:
    stat = ImageStat.Stat(img.convert("L"))
    return stat.mean[0]

def variance_of_laplacian_numpy(img: Image.Image) -> float:
    # Fallback if OpenCV not available: approximate Laplacian variance using numpy gradients
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    return float(np.var(grad_mag))

def image_blur_score(img_bytes: bytes) -> float:
    """
    Returns a blur score: higher = sharper. Uses OpenCV if available, else numpy fallback.
    Thresholds:
      - > 100 : sharp
      - 30-100 : acceptable
      - < 30 : blurry
    """
    try:
        import cv2
        import numpy as np
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        lap = cv2.Laplacian(img, cv2.CV_64F)
        var = float(lap.var())
        return var
    except Exception:
        # fallback
        try:
            img = Image.open(__import__("io").BytesIO(img_bytes))
            return variance_of_laplacian_numpy(img)
        except Exception:
            return 0.0

def image_too_small(img_bytes: bytes, min_pixels: int = 224*224) -> bool:
    try:
        img = Image.open(__import__("io").BytesIO(img_bytes))
        w, h = img.size
        return (w * h) < min_pixels
    except Exception:
        return True

def image_quality_check(img_bytes: bytes) -> Dict[str, Any]:
    """
    Returns dict with keys:
      - ok: bool (True if image is acceptable)
      - reasons: list[str]
      - blur_score: float
      - brightness: float
      - size_ok: bool
    """
    reasons = []
    blur = image_blur_score(img_bytes)
    size_ok = not image_too_small(img_bytes)
    brightness = None
    try:
        img = Image.open(__import__("io").BytesIO(img_bytes))
        brightness = image_brightness(img)
    except Exception:
        brightness = 0.0

    # thresholds tuned for production conservative behavior
    if blur < 30:
        reasons.append("image_blurry")
    if not size_ok:
        reasons.append("image_too_small")
    if brightness is not None and brightness < 20:
        reasons.append("image_too_dark")
    ok = len(reasons) == 0
    return {
        "ok": ok,
        "reasons": reasons,
        "blur_score": blur,
        "brightness": brightness,
        "size_ok": size_ok
    }

# -------------------------
# URL validation (production)
# -------------------------
async def _fetch_text(client: httpx.AsyncClient, url: str, timeout: float = 6.0) -> Optional[str]:
    try:
        r = await client.get(url, timeout=timeout, follow_redirects=True)
        # only consider HTML responses
        ctype = r.headers.get("content-type", "")
        if "text/html" in ctype or "application/xhtml+xml" in ctype:
            return r.text[:200000]
        return ""
    except Exception:
        return None

@ttl_cache(_URL_CACHE_TTL)
async def validate_product_url(url: str, timeout: float = 6.0) -> Dict[str, Any]:
    """
    Production-grade validation:
      - rejects blacklisted patterns
      - fetches page (short timeout)
      - checks heuristics (price, sku, product microdata)
      - returns structured result with score and reason
    """
    if not is_valid_product_link(url):
        return {"url": url, "ok": False, "score": 0, "reason": "invalid_format_or_blacklisted"}

    domain = domain_from_url(url)
    base_score = 35 if domain in WHITELIST_DOMAINS else 10

    async with httpx.AsyncClient() as client:
        text = await _fetch_text(client, url, timeout=timeout)
        if text is None:
            return {"url": url, "ok": False, "score": 0, "reason": "fetch_error"}
        score = base_score
        if looks_like_product_page_text(text):
            score += 50
        if re.search(r'(\d[\d\s,.]{1,6})\s?(€|eur|€)', text, flags=re.IGNORECASE):
            score += 10
        if re.search(r'\b(réf|référence|sku|part ?no|partnumber)\b', text, flags=re.IGNORECASE):
            score += 10
        ok = score >= 50
        reason = "ok" if ok else "low_score"
        return {"url": url, "ok": ok, "score": score, "reason": reason, "domain": domain}

# -------------------------
# Perplexity / Sonar call (production)
# -------------------------
async def call_perplexity_api(query: str, max_candidates: int = 8) -> Any:
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        return {"error": "PERPLEXITY_API_KEY_MISSING"}

    system_content = """
    <system_instructions>
    <role>Expert en Sourcing Industriel B2B/B2C</role>
    <task>Retourner UNIQUEMENT des liens directs vers fiches produits (SKU individuels).</task>
    <constraints>
      <constraint>Vérifier dimensions exactes (filetage, diamètre)</constraint>
      <constraint>Retourner uniquement URLs de produits précis</constraint>
    </constraints>
    <output_format>JSON array: [{"nom":"", "prix":"", "url":"", "source":""}]</output_format>
    </system_instructions>
    """

    data = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Trouve produits pour : {query} Donne max {max_candidates} résultats, format JSON."}
        ],
        "temperature": 0.1
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    attempt = 0
    while attempt <= PERPLEXITY_RETRIES:
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(PERPLEXITY_API_URL, json=data, headers=headers, timeout=PERPLEXITY_TIMEOUT)
            if res.status_code != 200:
                attempt += 1
                await asyncio.sleep(PERPLEXITY_BACKOFF ** attempt)
                continue
            payload = res.json()
            # Expecting the model to return JSON in choices[0].message.content
            raw = payload.get("choices", [{}])[0].get("message", {}).get("content")
            if not raw:
                return {"error": "empty_response"}
            parsed = json.loads(raw)
            # parsed may be {"produits": [...] } or a list
            if isinstance(parsed, dict) and "produits" in parsed:
                candidates = parsed["produits"]
            elif isinstance(parsed, list):
                candidates = parsed
            else:
                # try to be tolerant
                candidates = parsed if isinstance(parsed, list) else []
            return {"candidates": candidates}
        except Exception as e:
            attempt += 1
            if attempt > PERPLEXITY_RETRIES:
                return {"error": f"perplexity_error:{str(e)[:200]}"}
            await asyncio.sleep(PERPLEXITY_BACKOFF ** attempt)
    return {"error": "perplexity_failed"}

# -------------------------
# High-level pipeline
# -------------------------
async def search_perplexity_async(query: str, max_candidates: int = 8) -> List[Dict[str, Any]]:
    """
    Production-ready wrapper:
      - calls Perplexity/Sonar
      - validates candidate URLs in parallel
      - returns filtered, scored list (may return empty list)
    """
    if not query or not isinstance(query, str) or query.strip() == "":
        return []

    resp = await call_perplexity_api(query, max_candidates=max_candidates)
    if "error" in resp:
        return [{"error": resp["error"]}]

    candidates = resp.get("candidates", [])[:max_candidates]
    # ensure structure
    normalized = []
    for c in candidates:
        if isinstance(c, dict) and c.get("url"):
            normalized.append({
                "nom": c.get("nom") or "",
                "prix": c.get("prix") or "",
                "url": c.get("url"),
                "source": c.get("source") or domain_from_url(c.get("url", ""))
            })

    # validate in parallel with concurrency limit
    results = []
    sem = asyncio.Semaphore(8)

    async def _validate_item(item):
        async with sem:
            v = await validate_product_url(item["url"])
            merged = {
                "nom": item.get("nom"),
                "prix": item.get("prix"),
                "url": item.get("url"),
                "source": item.get("source"),
                "valid": v.get("ok", False),
                "score": v.get("score", 0),
                "reason": v.get("reason", "")
            }
            return merged

    tasks = [_validate_item(it) for it in normalized]
    if tasks:
        validated = await asyncio.gather(*tasks, return_exceptions=False)
    else:
        validated = []

    # sort by score desc, prefer valid ones
    validated_sorted = sorted(validated, key=lambda x: (1 if x["valid"] else 0, x["score"]), reverse=True)
    # return only valid first; if none valid, return top candidates (so UX can show them)
    valid = [v for v in validated_sorted if v["valid"]]
    return valid if valid else validated_sorted

# -------------------------
# HTML rendering helper
# -------------------------
def format_links_html(products: List[Dict[str, Any]]) -> str:
    if not products:
        return "Aucune fiche produit directe trouvée."
    html = ""
    for p in products:
        url = p.get("url")
        name = p.get("nom") or url
        price = p.get("prix") or "Voir prix"
        score = p.get("score", 0)
        domain = p.get("source") or domain_from_url(url)
        html += f'<a href="{url}" target="_blank" class="buy-link">🛒 {name} - {price} <small style="opacity:.6">[{domain} • {score}]</small></a>'
    return html

# -------------------------
# Endpoint: identify (keeps HTML intact)
# -------------------------
@app.post("/identify", response_class=HTMLResponse)
async def identify(image: UploadFile = File(...), context: str = Form("")):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    try:
        raw_bytes = await image.read()

        # 1) Quality checks: blur, size, brightness
        quality = image_quality_check(raw_bytes)
        if not quality["ok"]:
            # Provide explicit reasons and do not call sourcing if image is poor
            reasons_map = {
                "image_blurry": "Image floue ou manque de netteté",
                "image_too_small": "Image trop petite / faible résolution",
                "image_too_dark": "Image trop sombre"
            }
            reasons_text = ", ".join(reasons_map.get(r, r) for r in quality["reasons"])
            return f"""
            <div class="results animate-in">
                <div class="res-card" style="color:#b91c1c"><strong>⚠️ Qualité image insuffisante</strong>
                <p>La photo fournie semble inadaptée pour un sourcing fiable : {reasons_text}.</p>
                <p>Score netteté: {quality['blur_score']:.1f} • Luminosité: {quality['brightness']:.1f}</p>
                </div>
                <div class="res-card shop"><strong>🔗 Fiches Produits Directes</strong><div class="links-list">Aucun résultat — améliore la photo et réessaie.</div></div>
                <button class="btn btn-run" onclick="location.reload()">🔄 Nouveau Diagnostic</button>
            </div>
            """

        # 2) Call Groq model to extract structured technical info
        img_b64 = base64.b64encode(raw_bytes).decode('utf-8')
        prompt = "ID TECHNIQUE. Format JSON: {\"mat\": \"\", \"std\": \"\", \"search\": \"\"}"

        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": [{"type": "text", "text": f"{prompt} Context: {context}"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}],
            response_format={"type": "json_object"}
        )

        # Parse model output safely
        try:
            data = json.loads(completion.choices[0].message.content)
        except Exception:
            # If parsing fails, return a clear error
            return f"<div class='res-card' style='color:red'>Erreur: réponse du modèle illisible.</div>"

        # If model indicates low confidence or asks for clearer image, respect it
        # We expect model to optionally include a 'confidence' or 'note' field; be defensive
        model_note = data.get("note") or ""
        model_confidence = data.get("confidence")  # optional numeric
        if isinstance(model_confidence, (int, float)) and model_confidence < 0.4:
            return f"""
            <div class="results animate-in">
                <div class="res-card" style="color:#b91c1c"><strong>⚠️ Confiance modèle faible</strong>
                <p>Le modèle indique une faible confiance ({model_confidence:.2f}) pour l'identification. {model_note}</p>
                </div>
                <div class="res-card shop"><strong>🔗 Fiches Produits Directes</strong><div class="links-list">Aucun résultat — fournis une photo plus nette ou plus d'angles.</div></div>
                <button class="btn btn-run" onclick="location.reload()">🔄 Nouveau Diagnostic</button>
            </div>
            """

        # 3) Build search query from model output
        search_query = data.get("search") or ""
        # If model didn't provide a search query, try to build one from mat/std
        if not search_query:
            parts = []
            if data.get("mat"):
                parts.append(data.get("mat"))
            if data.get("std"):
                parts.append(data.get("std"))
            search_query = " ".join(parts).strip()
        if not search_query:
            return f"<div class='res-card' style='color:red'>Erreur: aucun terme de recherche généré par le modèle.</div>"

        # 4) Call Perplexity/Sonar to get candidate product URLs and validate them
        candidates = await search_perplexity_async(search_query)

        # 5) Format links for HTML
        links_html = format_links_html(candidates)

        # 6) Return the same HTML structure as before, injecting results
        return f"""
        <div class="results animate-in">
            <div class="res-card mat"><strong>🧪 Matière</strong><p>{data.get('mat')}</p></div>
            <div class="res-card std"><strong>📏 Technique</strong><p>{data.get('std')}</p></div>
            <div class="res-card shop"><strong>🔗 Fiches Produits Directes</strong><div class="links-list">{links_html}</div></div>
            <button class="btn btn-run" onclick="location.reload()">🔄 Nouveau Diagnostic</button>
        </div>
        """
    except Exception as e:
        return f"<div class='res-card' style='color:red'>Erreur Vision : {str(e)}</div>"

# -------------------------
# Home route (HTML intact)
# -------------------------
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
