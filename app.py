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
import streamlit.components.v1 as components

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
    """Decodes browser WebM/Ogg/WAV bytes into 16kHz mono float32 PCM via PyAV without metadata_errors."""
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
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    messages = [{"role": "system", "content": system_prompt}] + conversation[-10:]
    logs = []
    url_to_open = ""

    try:
        for _ in range(4):
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                final_text = msg.content or "Done."
                if url_to_open and url_to_open not in final_text:
                    final_text += f"\n\n🔗 [Open {url_to_open}]({url_to_open})"
                return final_text, logs, url_to_open

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                fn = tc.function.name
                args = json.loads(tc.function.arguments or "{}")

                if fn == "web_research":
                    team, out = "Research Team", web_research(**args)
                elif fn == "build_or_update_website":
                    team = "Dev Team (Background)"
                    out, _ = spawn_background_website_builder(
                        api_key,
                        model_id,
                        args.get("project_title", "Landing Page"),
                        args.get("requirements", ""),
                    )
                elif fn == "open_browser_or_youtube":
                    team = "Browser Control"
                    resolved = resolve_browser_url(args.get("target", ""))
                    url_to_open = resolved
                    out = f"Resolved client browser URL: {resolved}"
                elif fn == "organize_windows_folder":
                    team, out = "Operations Team", organize_windows_folder(**args)
                elif fn == "trigger_n8n_operations":
                    team = "Ops / n8n"
                    out = trigger_n8n_operations(
                        n8n_url,
                        args.get("task_summary", ""),
                        args.get("payload_json", "{}"),
                    )
                else:
                    team, out = "System", f"Unknown tool: {fn}"

                logs.append({"team": team, "tool": fn, "args": args, "output": out})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": fn,
                        "content": str(out),
                    }
                )

        return "Completed multi-step workflow.", logs, url_to_open
    except RateLimitError:
        return "OpenRouter free rate limit reached. Please wait a moment or switch models in the sidebar.", logs, ""
    except Exception as e:
        return f"API error: {e}", logs, ""


# ---------------------------------------------------------
# 4. Hands-Free VAD Avatar Component (AnalyserNode + MediaRecorder)
# ---------------------------------------------------------
def render_handsfree_hud_component(
    agent_name: str,
    hands_free_on: bool,
    reply_audio_b64: str,
    open_url: str,
    turn_id: int,
):
    render_nonce = int(time.time() * 1000)
    js_agent_name = json.dumps(agent_name)
    js_hands_free = "true" if hands_free_on else "false"
    js_reply_b64 = json.dumps(reply_audio_b64 or "")
    js_open_url = json.dumps(open_url or "")

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <!-- turn:{turn_id} nonce:{render_nonce} -->
    <head>
    <style>
      body {{
        margin: 0;
        background: transparent;
        font-family: 'Segoe UI', sans-serif;
        color: #f1f5f9;
      }}
      .hud-card {{
        background: rgba(15, 23, 42, 0.96);
        border: 1px solid rgba(56, 189, 248, 0.55);
        border-radius: 18px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
      }}
      .left-group {{ display: flex; align-items: center; gap: 16px; }}
      .avatar-orb {{
        width: 68px; height: 68px; border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #0284c7, #0f172a);
        border: 2px solid #e2e8f0;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.6);
        transition: all 0.25s ease;
        cursor: pointer;
      }}
      .avatar-orb.listening {{
        background: radial-gradient(circle at 35% 35%, #16a34a, #0f172a);
        box-shadow: 0 0 24px rgba(74, 222, 128, 0.85);
        animation: pulse 1.2s infinite;
      }}
      .avatar-orb.speaking {{
        background: radial-gradient(circle at 35% 35%, #7e22ce, #0f172a);
        box-shadow: 0 0 24px rgba(192, 132, 252, 0.85);
        animation: pulse 0.7s infinite;
      }}
      @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.06); }}
        100% {{ transform: scale(1); }}
      }}
      .eyes {{ display: flex; gap: 10px; }}
      .eye {{ width: 7px; height: 7px; background: #f8fafc; border-radius: 50%; }}
      .title {{ font-size: 18px; font-weight: 700; margin: 0 0 4px 0; }}
      .status {{ font-size: 13px; color: #38bdf8; margin: 0; }}
      .meter-row {{ display: flex; align-items: center; gap: 8px; margin-top: 6px; }}
      .meter-bar {{
        width: 160px; height: 6px; background: #1e293b;
        border-radius: 4px; overflow: hidden;
      }}
      .meter-fill {{ height: 100%; width: 0%; background: #4ade80; transition: width 0.06s linear; }}
      .lvl-txt {{ font-size: 11px; color: #94a3b8; font-family: monospace; }}
      .send-btn {{
        background: #0284c7; color: #fff; border: none; border-radius: 10px;
        padding: 8px 12px; font-size: 12px; font-weight: 600; cursor: pointer;
      }}
      .send-btn:hover {{ background: #0369a1; }}
    </style>
    </head>
    <body>
      <div class="hud-card">
        <div class="left-group">
          <div id="orb" class="avatar-orb" title="Click orb to send immediately">
            <div class="eyes"><div class="eye"></div><div class="eye"></div></div>
          </div>
          <div>
            <div class="title" id="titleBox"></div>
            <div id="statusText" class="status">● Initializing...</div>
            <div class="meter-row">
              <div class="meter-bar"><div id="meterFill" class="meter-fill"></div></div>
              <span id="lvlTxt" class="lvl-txt">lvl: 0</span>
            </div>
          </div>
        </div>
        <button id="sendNowBtn" class="send-btn" style="display:none;">Send Now ↵</button>
        <audio id="ttsPlayer"></audio>
      </div>

      <script>
        const agentName = {js_agent_name};
        const handsFree = {js_hands_free};
        const replyB64 = {js_reply_b64};
        const pendingUrl = {js_open_url};
        const turnId = {turn_id};

        const orb = document.getElementById("orb");
        const titleBox = document.getElementById("titleBox");
        const statusText = document.getElementById("statusText");
        const meterFill = document.getElementById("meterFill");
        const lvlTxt = document.getElementById("lvlTxt");
        const sendNowBtn = document.getElementById("sendNowBtn");
        const ttsPlayer = document.getElementById("ttsPlayer");

        let recordingActive = false;
        let forceStopFn = null;

        titleBox.innerHTML = agentName + ' <span style="font-size:11px;color:#94a3b8;">(v3 • Turn ' + (turnId + 1) + ')</span>';

        if (pendingUrl && pendingUrl.startsWith("http")) {{
          try {{
            window.top.open(pendingUrl, "_blank", "noopener,noreferrer");
          }} catch (e) {{
            window.open(pendingUrl, "_blank", "noopener,noreferrer");
          }}
        }}

        function sendAudioToStreamlit(base64Audio) {{
          const parentDoc = window.parent.document;
          const inputs = parentDoc.querySelectorAll('input[aria-label="vad_audio_payload"]');
          if (inputs.length > 0) {{
            const targetInput = inputs[0];
            const lastValue = targetInput.value;
            const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
            nativeSetter.call(targetInput, turnId + "::" + base64Audio);

            const tracker = targetInput._valueTracker;
            if (tracker) tracker.setValue(lastValue);

            targetInput.dispatchEvent(new Event("input", {{ bubbles: true }}));
            targetInput.dispatchEvent(new Event("change", {{ bubbles: true }}));
            setTimeout(() => {{
              targetInput.dispatchEvent(new KeyboardEvent("keydown", {{
                key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true
              }}));
            }}, 100);
          }} else {{
            statusText.style.color = "#f87171";
            statusText.innerText = "● Bridge input not found";
          }}
        }}

        async function startAutoSilenceRecording() {{
          if (!handsFree || recordingActive) return;
          recordingActive = true;
          try {{
            statusText.style.color = "#facc15";
            statusText.innerText = "● Opening microphone...";
            const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            if (audioCtx.state === "suspended") await audioCtx.resume();

            const source = audioCtx.createMediaStreamSource(stream);
            const analyser = audioCtx.createAnalyser();
            analyser.fftSize = 1024;
            source.connect(analyser);

            const mediaRecorder = new MediaRecorder(stream);
            const chunks = [];
            mediaRecorder.ondataavailable = (e) => {{
              if (e.data && e.data.size > 0) chunks.push(e.data);
            }};

     let speechStarted = false;
            let silenceStart = null;
            let stopped = false;
            let manualForced = false;

            // Auto-calibrate room background noise for the first 600ms
            let calibSamples = [];
            let noiseFloor = 8;
            let speechThreshold = 15;
            const SILENCE_DURATION_MS = 1400;
            const MAX_RECORD_MS = 15000;
            const startTime = Date.now();

            orb.className = "avatar-orb listening";
            statusText.style.color = "#facc15";
            statusText.innerText = "● Calibrating room noise (0.6s)...";
            sendNowBtn.style.display = "inline-block";

            function stopAndSubmit(isManual) {{
              if (stopped) return;
              stopped = true;
              manualForced = isManual;
              sendNowBtn.style.display = "none";
              if (mediaRecorder.state !== "inactive") {{
                mediaRecorder.stop();
              }}
            }}

            forceStopFn = () => stopAndSubmit(true);
            sendNowBtn.onclick = () => stopAndSubmit(true);

            mediaRecorder.onstop = () => {{
              recordingActive = false;
              forceStopFn = null;
              stream.getTracks().forEach(t => t.stop());
              audioCtx.close();
              meterFill.style.width = "0%";
              lvlTxt.innerText = "lvl: 0";

              if (!speechStarted && !manualForced) {{
                if (handsFree) setTimeout(startAutoSilenceRecording, 250);
                return;
              }}

              orb.className = "avatar-orb";
              statusText.style.color = "#facc15";
              statusText.innerText = "● Transcribing your voice...";

              const blob = new Blob(chunks, {{ type: mediaRecorder.mimeType || "audio/webm" }});
              const reader = new FileReader();
              reader.onloadend = () => {{
                const b64 = reader.result.split(",")[1];
                sendAudioToStreamlit(b64);
              }};
              reader.readAsDataURL(blob);
            }};

            mediaRecorder.start(250);

            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            function monitorVolume() {{
              if (stopped) return;
              analyser.getByteFrequencyData(dataArray);
              let sum = 0;
              for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
              const avgLevel = Math.round((sum / dataArray.length) / 2.55); // 0 to 100
              const elapsed = Date.now() - startTime;

              meterFill.style.width = Math.min(100, avgLevel * 2) + "%";
              lvlTxt.innerText = "lvl: " + avgLevel + " / thr: " + speechThreshold;

              // Phase 1: Measure room ambient noise for 600ms
              if (elapsed < 600) {{
                calibSamples.push(avgLevel);
                requestAnimationFrame(monitorVolume);
                return;
              }} else if (calibSamples.length > 0) {{
                const avgNoise = calibSamples.reduce((a, b) => a + b, 0) / calibSamples.length;
                noiseFloor = Math.round(avgNoise);
                speechThreshold = Math.min(45, Math.max(noiseFloor + 6, 12));
                calibSamples = [];
                statusText.style.color = "#4ade80";
                statusText.innerText = "● Listening (Speak now — auto-sends on pause)...";
              }}

              // Phase 2: Detect speech above calibrated room noise & auto-stop on pause
              if (avgLevel >= speechThreshold) {{
                if (!speechStarted) {{
                  speechStarted = true;
                  statusText.innerText = "● Hearing you (auto-sends when you pause)...";
                }}
                silenceStart = null;
              }} else if (speechStarted) {{
                if (!silenceStart) silenceStart = Date.now();
                if (Date.now() - silenceStart >= SILENCE_DURATION_MS) {{
                  stopAndSubmit(false);
                  return;
                }}
              }}

              if (elapsed >= MAX_RECORD_MS) {{
                stopAndSubmit(false);
                return;
              }}
              requestAnimationFrame(monitorVolume);
            }}
            function monitorVolume() {{
              if (stopped) return;
              analyser.getByteFrequencyData(dataArray);
              let sum = 0;
              for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
              const avgLevel = Math.round((sum / dataArray.length) / 2.55); // 0 to 100

              meterFill.style.width = Math.min(100, avgLevel * 2) + "%";
              lvlTxt.innerText = "lvl: " + avgLevel;

              if (avgLevel >= SPEECH_THRESHOLD) {{
                if (!speechStarted) {{
                  speechStarted = true;
                  statusText.innerText = "● Hearing you (auto-sends when you pause)...";
                }}
                silenceStart = null;
              }} else if (speechStarted) {{
                if (!silenceStart) silenceStart = Date.now();
                if (Date.now() - silenceStart >= SILENCE_DURATION_MS) {{
                  stopAndSubmit(false);
                  return;
                }}
              }}

              if (Date.now() - startTime >= MAX_RECORD_MS) {{
                stopAndSubmit(false);
                return;
              }}
              requestAnimationFrame(monitorVolume);
            }}
            monitorVolume();
          }} catch (err) {{
            recordingActive = false;
            sendNowBtn.style.display = "none";
            statusText.style.color = "#f87171";
            statusText.innerText = "● Mic notice: " + (err.message || err);
          }}
        }}

        orb.addEventListener("click", () => {{
          if (recordingActive && forceStopFn) {{
            forceStopFn();
          }} else if (handsFree && !recordingActive) {{
            startAutoSilenceRecording();
          }}
        }});

        function resumeAfterSpeech() {{
          orb.className = "avatar-orb";
          statusText.style.color = "#38bdf8";
          statusText.innerText = "● Standby";
          if (handsFree) setTimeout(startAutoSilenceRecording, 350);
        }}

        if (!handsFree) {{
          orb.className = "avatar-orb";
          statusText.style.color = "#38bdf8";
          statusText.innerText = "● Standby — Click Start Live to speak";
        }} else if (replyB64 && replyB64.length > 10) {{
          orb.className = "avatar-orb speaking";
          statusText.style.color = "#c084fc";
          statusText.innerText = "● Speaking...";
          ttsPlayer.src = "data:audio/mp3;base64," + replyB64;

          let resumed = false;
          const safeResume = () => {{
            if (resumed) return;
            resumed = true;
            resumeAfterSpeech();
          }};

          ttsPlayer.onended = safeResume;
          ttsPlayer.onerror = safeResume;
          ttsPlayer.play().catch(safeResume);

          setTimeout(() => {{
            if (ttsPlayer.paused && !resumed) safeResume();
          }}, 12000);
        }} else {{
          startAutoSilenceRecording();
        }}
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=116)


# ---------------------------------------------------------
# 5. Main Streamlit App UI (Desktop HUD Theme)
# ---------------------------------------------------------
st.set_page_config(page_title="M.I.T.H.U. Hands-Free Voice Agent", page_icon="🎙️", layout="centered")

st.markdown(
    """
    <style>
      .stApp { background-color: #0b1120; color: #f1f5f9; }
      div[data-testid="stTextInput"]:has(input[aria-label="vad_audio_payload"]) {
        position: fixed; left: -9999px; opacity: 0; height: 0; overflow: hidden;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🎙️ Personalize Agent")
    agent_name = st.text_input("Companion Name", value="M.I.T.H.U.")
    user_name = st.text_input("Your Name", value="Tanmoy")
    business_context = st.text_input(
        "Role / Business Context",
        value="Software development, AI automation, and digital education",
    )
    language_pref = st.selectbox(
        "Conversation Style",
        ["English + Bengali (Bilingual)", "English", "Bengali", "Hindi"],
    )
    voice_label = st.selectbox("Neural Voice Output", list(VOICE_MAP.keys()))
    selected_voice = VOICE_MAP[voice_label]
    voice_enabled = st.toggle("🔊 Speak Responses Aloud", value=True)

    st.divider()
    default_key = get_default_api_key()
    if default_key:
        api_key = default_key
        st.success("🔑 API Key loaded automatically")
    else:
        api_key = st.text_input(
            "OpenRouter API Key",
            value="",
            type="password",
            help="Enter your OpenRouter API key to start.",
        )

    model_id = st.selectbox(
        "Nemotron Model",
        [
            "nvidia/nemotron-3.5-lightning:free",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "openrouter/free",
        ],
    )
    n8n_url = st.text_input("n8n Webhook URL", value="http://localhost:5678/webhook/agent-ops")

    generated_files = sorted(WORKSPACE_DIR.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    if generated_files:
        st.divider()
        st.subheader("💻 Dev Team Builds")
        for f in generated_files:
            st.download_button(
                label=f"⬇️ Download {f.name}",
                data=f.read_bytes(),
                file_name=f.name,
                mime="text/html",
                key=f"dl_{f.name}",
                use_container_width=True,
            )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.reply_audio_b64 = ""
        st.session_state.pending_open_url = ""
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "hands_free_on" not in st.session_state:
    st.session_state.hands_free_on = False
if "reply_audio_b64" not in st.session_state:
    st.session_state.reply_audio_b64 = ""
if "pending_open_url" not in st.session_state:
    st.session_state.pending_open_url = ""
if "turn_id" not in st.session_state:
    st.session_state.turn_id = 0
if "last_processed_turn" not in st.session_state:
    st.session_state.last_processed_turn = -1

# Top Control Bar
c1, c2 = st.columns([3, 1])
with c1:
    st.caption("🔍 Research • ✍️ Content • 📣 Ads • 💻 Dev (Background) • 🎧 Support • ⚙️ Ops")
with c2:
    btn_label = "🔴 Stop Live" if st.session_state.hands_free_on else "🎤 Start Live"
    if st.button(btn_label, use_container_width=True, type="primary"):
        st.session_state.hands_free_on = not st.session_state.hands_free_on
        st.session_state.reply_audio_b64 = ""
        st.session_state.pending_open_url = ""
        st.rerun()

# Render Glowing HUD + Hands-Free Auto-Stop Voice Loop + Client Tab Opener
render_handsfree_hud_component(
    agent_name=agent_name,
    hands_free_on=st.session_state.hands_free_on,
    reply_audio_b64=st.session_state.reply_audio_b64,
    open_url=st.session_state.pending_open_url,
    turn_id=st.session_state.turn_id,
)
st.session_state.reply_audio_b64 = ""
st.session_state.pending_open_url = ""

# Hidden bridge input receiving base64 browser audio from VAD
vad_payload = st.text_input("vad_audio_payload", key="vad_bridge")

# Display Transcript
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("logs"):
            with st.expander("🛠️ Specialist Team Activity"):
                for log in m["logs"]:
                    st.markdown(f"**[{log['team']}]** `{log['tool']}` → `{log['args']}`")
                    st.code(log["output"][:800])

typed_prompt = st.chat_input(f"Or type a command for {agent_name}...")
active_prompt = None

if vad_payload and "::" in vad_payload:
    turn_str, b64_audio = vad_payload.split("::", 1)
    try:
        incoming_turn = int(turn_str)
    except ValueError:
        incoming_turn = -1

    if incoming_turn == st.session_state.turn_id and incoming_turn != st.session_state.last_processed_turn:
        st.session_state.last_processed_turn = incoming_turn
        with st.spinner("Transcribing speech..."):
            raw_bytes = base64.b64decode(b64_audio)
            audio_float32 = decode_audio_bytes_to_16k_float32(raw_bytes)
            if audio_float32.size > 0:
                whisper = load_whisper("base")
                segments, _ = whisper.transcribe(audio_float32, beam_size=1)
                transcribed = " ".join(seg.text for seg in segments).strip()
            else:
                transcribed = ""

            if transcribed:
                active_prompt = transcribed
            else:
                st.session_state.turn_id += 1
                st.rerun()

if typed_prompt:
    active_prompt = typed_prompt

if active_prompt:
    if not api_key:
        st.error("OpenRouter API Key not found. Add OPENROUTER_API_KEY to .streamlit/secrets.toml or Streamlit Cloud Secrets.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": active_prompt})
    clean_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    reply, logs, url_to_open = run_orchestrator(
        api_key=api_key,
        model_id=model_id,
        agent_name=agent_name,
        user_name=user_name,
        business_context=business_context,
        language_pref=language_pref,
        n8n_url=n8n_url,
        conversation=clean_history,
    )

    if voice_enabled:
        st.session_state.reply_audio_b64 = synthesize_speech_b64(reply, selected_voice)
    if url_to_open:
        st.session_state.pending_open_url = url_to_open

    st.session_state.messages.append({"role": "assistant", "content": reply, "logs": logs})
    st.session_state.turn_id += 1
    st.rerun()