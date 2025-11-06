# app.py — OpenAI Responses API + OMDb poster lookup (Chanel / Netflix style)
import os
import json
import re
import requests
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

app = Flask(__name__, static_folder="static", template_folder="templates")

# --- ENVIRONMENT VARIABLES ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")  # <--- You must set this
if not OPENAI_API_KEY:
    raise RuntimeError("Please set your OPENAI_API_KEY in the environment.")
if not OMDB_API_KEY:
    print("⚠️ OMDb key not found — posters will not load.")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- HELPER FUNCTIONS ---

def build_prompt(mood: str) -> str:
    return (
        f"You are a helpful movie curator. Based on the user's mood '{mood}', "
        f"suggest 3 movies and 3 songs.\n"
        "Output a valid JSON object with keys 'movies' and 'songs'.\n"
        "Each movie must have: title, year (if known), why (1 sentence).\n"
        "Each song must have: title, artist (if known), why (1 sentence).\n"
        "Return strictly JSON, no other text.\n"
        "Example:\n"
        "{"
        "\"movies\": ["
        "{\"title\": \"In the Mood for Love\", \"year\": \"2000\", \"why\": \"A melancholic and visually poetic film about longing.\"}"
        "],"
        "\"songs\": ["
        "{\"title\": \"The Look of Love\", \"artist\": \"Dusty Springfield\", \"why\": \"Smooth and romantic.\"}"
        "]}"
    )


def extract_text_from_response(response):
    """Handle both new Responses API and fallback cases."""
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    output = getattr(response, "output", None)
    if output:
        parts = []
        for block in output:
            if isinstance(block, dict):
                content = block.get("content")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, str):
                            parts.append(c)
                        elif isinstance(c, dict) and "text" in c:
                            parts.append(c["text"])
                elif isinstance(content, str):
                    parts.append(content)
        return "".join(parts)
    return str(response)


def parse_json_block(text: str):
    """Find and load JSON block safely."""
    text = text.replace("“", "\"").replace("”", "\"").replace("’", "'")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except Exception:
        return {"raw_text": text}


def omdb_lookup(title: str, year: str = None):
    """Fetch poster + metadata from OMDb."""
    if not OMDB_API_KEY:
        return None, None, None, None
    try:
        params = {"t": title, "apikey": OMDB_API_KEY}
        if year:
            params["y"] = year
        res = requests.get("https://www.omdbapi.com/", params=params, timeout=8)
        data = res.json()
        if data.get("Response") == "True":
            return (
                data.get("Poster") if data.get("Poster") != "N/A" else None,
                data.get("Year"),
                data.get("Genre"),
                data.get("Plot"),
            )
    except Exception:
        pass
    return None, None, None, None


# --- ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json() or {}
    mood = data.get("mood")
    if not mood:
        return jsonify({"error": "Mood is required"}), 400

    prompt = build_prompt(mood)

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            temperature=0.7,
            max_output_tokens=700,
        )
        text = extract_text_from_response(response)
        parsed = parse_json_block(text)

        if "movies" not in parsed:
            return jsonify({"error": "No movie data returned", "raw": text})

        # Enrich movies with OMDb metadata
        enriched = []
        for m in parsed["movies"]:
            title = m.get("title")
            year = m.get("year")
            why = m.get("why", "")
            poster, real_year, genre, plot = omdb_lookup(title, year)
            enriched.append({
                "title": title,
                "year": real_year or year,
                "why": why,
                "poster_url": poster or "/static/poster-placeholder.png",
                "genre": genre,
                "plot": plot,
            })
        parsed["movies"] = enriched
        return jsonify({"recommendations": parsed})

    except Exception as e:
        return jsonify({"error": "OpenAI request failed", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
