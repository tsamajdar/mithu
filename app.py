import os
import io
import re
import json
import time
import base64
import asyncio
import shutil
import tempfile
import threading
from pathlib import Path

import av
import numpy as np
import requests
from ddgs import DDGS
from openai import OpenAI, RateLimitError
import edge_tts
from faster_whisper import WhisperModel
import streamlit as st

# ---------------------------------------------------------
# 1. Configuration, Secrets & Voices
# ---------------------------------------------------------
VOICE_MAP = {
    "English (India - Neerja)": "en-IN-NeerjaNeural",
    "Bengali (India - Tanishaa)": "bn-IN-TanishaaNeural",
    "Bengali (Bangladesh - Nabanita)": "bn-BD-NabanitaNeural",
    "Hindi (India - Swara)": "hi-IN-SwaraNeural",
    "English (US - Aria)": "en-US-AriaNeural",
}

WORKSPACE_DIR = Path(tempfile.gettempdir()) / "ai_companion_workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


def get_default_api_key() -> str:
    """Automatically loads OPENROUTER_API_KEY from st.secrets or environment variables."""
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass
    return os.environ.get("OPENROUTER_API_KEY", "")


@st.cache_resource
def load_whisper(model_size: str = "base"):
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def decode_audio_bytes_to_16k_float32(audio_bytes: bytes) -> np.ndarray:
    """Decodes audio bytes into 16kHz mono float32 PCM via PyAV."""
    container = av.open(io.BytesIO(audio_bytes))
    resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=16000)
    chunks = []
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            arr = resampled.to_ndarray()
            chunks.append(arr.reshape(-1))
    container.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def synthesize_speech_b64(text: str, voice: str) -> str:
    """Generates base64 MP3 audio using Microsoft Edge Neural TTS."""
    clean_text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    clean_text = clean_text.replace("*", "").replace("#", "").replace("`", "").strip()
    if not clean_text:
        return ""

    tmp_mp3 = Path(tempfile.gettempdir()) / f"mithu_st_{int(time.time()*1000)}.mp3"
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        communicate = edge_tts.Communicate(clean_text[:900], voice)
        loop.run_until_complete(communicate.save(str(tmp_mp3)))
        loop.close()
        return base64.b64encode(tmp_mp3.read_bytes()).decode("ascii")
    finally:
        if tmp_mp3.exists():
            try:
                tmp_mp3.unlink()
            except Exception:
                pass


# ---------------------------------------------------------
# 2. Client-Side URL Resolver & Specialist Team Tools
# ---------------------------------------------------------
def resolve_browser_url(target: str) -> str:
    cleaned = target.strip()
    domain_candidate = re.sub(r"\s*\.\s*", ".", cleaned)
    if "." in domain_candidate and " " not in domain_candidate.replace("https://", "").replace("http://", ""):
        if not domain_candidate.startswith(("http://", "https://")):
            return f"https://{domain_candidate}"
        return domain_candidate

    if re.search(r"\.(com|in|co\.in|org|net|io|ai|app|edu|gov)\b", cleaned, re.IGNORECASE):
        compact = re.sub(r"\s+", "", cleaned)
        if not compact.startswith(("http://", "https://")):
            return f"https://{compact}"
        return compact

    query = requests.utils.quote(cleaned)
    return f"https://www.youtube.com/results?search_query={query}"


def web_research(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No live web results found."
        return "\n".join(
            f"- {r.get('title')}: {r.get('body')} ({r.get('href')})" for r in results
        )
    except Exception as e:
        return f"Research error: {e}"


def organize_windows_folder(folder_name: str = "Downloads") -> str:
    shortcuts = {
        "desktop": Path.home() / "Desktop",
        "downloads": Path.home() / "Downloads",
        "documents": Path.home() / "Documents",
    }
    target = shortcuts.get(folder_name.lower().strip(), Path(folder_name).expanduser())
    if not target.exists() or not target.is_dir():
        return f"Folder not found on host environment: {target}"

    categories = {
        "Images": {".jpg", ".jpeg", ".png", ".gif", ".webp"},
        "Documents": {".pdf", ".docx", ".doc", ".txt", ".xlsx", ".csv", ".pptx"},
        "Videos": {".mp4", ".mkv", ".mov"},
        "Archives": {".zip", ".rar", ".7z"},
        "Code": {".py", ".js", ".html", ".css", ".json", ".php"},
    }
    moved = {}
    for item in target.iterdir():
        if item.is_dir() or item.name.startswith("."):
            continue
        ext = item.suffix.lower()
        cat_name = next((c for c, exts in categories.items() if ext in exts), "Others")
        dest_dir = target / cat_name
        dest_dir.mkdir(exist_ok=True)
        dest_file = dest_dir / item.name
        if not dest_file.exists():
            shutil.move(str(item), str(dest_file))
            moved[cat_name] = moved.get(cat_name, 0) + 1

    if not moved:
        return f"No loose files needed sorting in {target}."
    return f"Organized {target}: " + ", ".join(f"{k} ({v})" for k, v in moved.items())


def trigger_n8n_operations(webhook_url: str, task_summary: str, payload_json: str = "{}") -> str:
    if not webhook_url.strip():
        return "No n8n webhook URL configured in the sidebar."
    try:
        resp = requests.post(
            webhook_url.strip(),
            json={"task": task_summary, "data": payload_json},
            timeout=6,
        )
        return f"Triggered n8n workflow (HTTP {resp.status_code})."
    except Exception as e:
        return f"Could not reach n8n ({webhook_url}): {e}"


def spawn_background_website_builder(
    api_key: str,
    model_id: str,
    project_title: str,
    requirements: str,
) -> tuple[str, Path]:
    safe_slug = "".join(c if c.isalnum() else "_" for c in project_title.lower()).strip("_")
    out_file = WORKSPACE_DIR / f"{safe_slug or 'landing_page'}.html"

    def _build_worker():
        try:
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            existing_code = out_file.read_text(encoding="utf-8") if out_file.exists() else ""
            prompt = (
                f"Build a complete, modern, single-file responsive HTML5 + Tailwind CSS (via CDN) webpage "
                f"for '{project_title}'. Requirements: {requirements}.\n"
                "Return ONLY raw valid HTML starting with <!DOCTYPE html>. No markdown fences."
            )
            if existing_code:
                prompt += f"\n\nUpgrade this existing HTML file with the requested changes:\n{existing_code[:6000]}"

            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            html_code = (resp.choices[0].message.content or "").replace("```html", "").replace("```", "").strip()
            out_file.write_text(html_code, encoding="utf-8")
        except Exception as e:
            (WORKSPACE_DIR / "build_error.log").write_text(str(e), encoding="utf-8")

    threading.Thread(target=_build_worker, daemon=True).start()
    return (
        f"Dev Team started building '{project_title}' in the background. It will appear in the sidebar once ready.",
        out_file,
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_research",
            "description": "[Research Team] Search live web facts, competitor info, or market prices.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_or_update_website",
            "description": "[Dev Team] Asynchronously build or upgrade an HTML/Tailwind landing page in the background.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_title": {"type": "string"},
                    "requirements": {"type": "string"},
                },
                "required": ["project_title", "requirements"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_browser_or_youtube",
            "description": "[Browser Control] Open a website URL (e.g. superweb.co.in) or search/play a song or video on YouTube.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "organize_windows_folder",
            "description": "[Operations Team] Sort loose files in Desktop, Downloads, or Documents into categorized folders.",
            "parameters": {
                "type": "object",
                "properties": {"folder_name": {"type": "string"}},
                "required": ["folder_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_n8n_operations",
            "description": "[Operations & Ads Team] Send a structured automation task to n8n.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_summary": {"type": "string"},
                    "payload_json": {"type": "string"},
                },
                "required": ["task_summary"],
            },
        },
    },
]


# ---------------------------------------------------------
# 3. Orchestrator Runner
# ---------------------------------------------------------
def run_orchestrator(
    api_key: str,
    model_id: str,
    agent_name: str,
    user_name: str,
    business_context: str,
    language_pref: str,
    n8n_url: str,
    conversation: list,
) -> tuple[str, list, str]:
    system_prompt = (
        f"You are {agent_name}, a warm, intelligent voice-first personal AI assistant for {user_name}. "
        f"Context: {business_context}. Preferred language style: {language_pref}. "
        "You coordinate 6 specialist teams: Research, Content, Ads, Dev, Support, and Operations. "
        "Keep spoken replies concise, clear, and conversational (1-3 sentences)."
    )
    client = OpenAI(base_url="