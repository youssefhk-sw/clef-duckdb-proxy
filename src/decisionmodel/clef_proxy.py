import asyncio
import json
import logging
import os
import sys
import time
from collections import deque, Counter
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import httpx
import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ! Real environment variables always win over .env values.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
load_dotenv(find_dotenv(usecwd=True))

# --------------------------------------------------------------------------
# 0. Rich logging (console + uvicorn/httpx/python logging all go through it)
# --------------------------------------------------------------------------
console = Console(log_path=False)
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
)

# --------------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------------
ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
ANALYTICS_TOKEN = os.getenv("CLOUDFLARE_ANALYTICS_TOKEN", "").strip() or API_TOKEN # One token can play the two tasks
MODEL = os.getenv("CLEF_MODEL", "clef-flash").strip()
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8787"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
DAILY_NEURON_LIMIT = float(os.getenv("DAILY_NEURON_LIMIT", "9500"))
USAGE_MIN_GAP = float(os.getenv("USAGE_MIN_GAP", "2"))
USAGE_FOLLOW_UP = sorted(float(x) for x in os.getenv("USAGE_FOLLOW_UP", "10,30,60,120,300").split(",") if x.strip())
USAGE_FOLLOW_UP_TEXT = "/".join(f"{x:g}" for x in USAGE_FOLLOW_UP) or "never"
STATE_TOKEN_WARN = int(os.getenv("STATE_TOKEN_WARN", "2000"))
TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT", "60"))
API_BASE = os.getenv("CF_API_BASE", "https://api.cloudflare.com/client/v4").rstrip("/")
USAGE_FILE = os.getenv("CLEF_USAGE_FILE", os.path.expanduser("~/.clef_proxy_usage.json"))

UPSTREAM_URL = f"{API_BASE}/accounts/{ACCOUNT_ID}/ai/run/@cf/cloudflare/{MODEL}"
GRAPHQL_URL = f"{API_BASE}/graphql"

FALLBACK_INSTRUCTIONS = "Answer this question about the given state."

MODELS = {
    "clef": {
        "id": "clef",
        "object": "model",
        "owned_by": "cloudflare",
        "name": "Clef",
        "provider_model_id": "@cf/cloudflare/clef",
        "description": "Higher-precision structured decision model.",
        "parameters": "27B",
        "context_window": 65536,
        "input_price_per_million_tokens": 0.24,
        "capabilities": ["noul", "choice", "score", "vision"],
        "last_verified_at": "2026-10-17",
        "source_url": (
            "https://developers.cloudflare.com/workers-ai/models/clef/"
        ),
    },
    "clef-flash": {
        "id": "clef-flash",
        "object": "model",
        "owned_by": "cloudflare",
        "name": "Clef-flash",
        "provider_model_id": "@cf/cloudflare/clef-flash",
        "description": "Faster decision model for latency-sensitive workloads.",
        "parameters": "9B",
        "context_window": 24576,
        "input_price_per_million_tokens": 0.038,
        "capabilities": ["noul", "choice", "score", "vision"],
        "last_verified_at": "2026-10-17",
        "source_url": (
            "https://developers.cloudflare.com/workers-ai/models/clef-flash/"
        ),
    },
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        headers={"Authorization": f"Bearer {API_TOKEN}", "User-Agent": "clef-duckdb-proxy/1"},
        timeout=httpx.Timeout(TIMEOUT, connect=10.0),
        limits=httpx.Limits(max_connections=MAX_CONCURRENCY, max_keepalive_connections=MAX_CONCURRENCY),
        transport=httpx.AsyncHTTPTransport(retries=2)
    )

    app.state.analytics_client = httpx.AsyncClient(
        headers={"Authorization": f"Bearer {ANALYTICS_TOKEN}", "User-Agent": "clef-duckdb-proxy/1"},
        timeout=httpx.Timeout(15.0, connect=10.0),
        transport=httpx.AsyncHTTPTransport(retries=1)
    )

    app.state.limiter = asyncio.Semaphore(MAX_CONCURRENCY)
    RT.dirty = asyncio.Event()
    await refresh_once(app) # today's real number before the first request
    refresher = asyncio.create_task(refresh_loop(app))
    yield
    refresher.cancel()
    if RT.settle:
        RT.settle.cancel()
    await asyncio.gather(refresher, return_exceptions=True)
    STORE.save(force=True)
    await app.state.client.aclose()
    await app.state.analytics_client.aclose()

app = FastAPI(title="clef-proxy", lifespan=lifespan)
    
# --------------------------------------------------------------------------
# 2. Helpers
# --------------------------------------------------------------------------
def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def new_day() -> dict:
    return {
        "requests": 0,            # requests forwarded to Cloudflare (answered or failed)
        "errors": 0,              
        "blocked": 0,             # refused by the budget guard (never sent)
        "questions": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "api_ms_total": 0,        # divide by `requests` for the average
        "api_ms_max": 0,
        "models_used": [],
        "cloudflare_neurons": None,
        "cloudflare_by_model": {},
        "neurons_read_at": None,
    }

def error_response(status: int, msg: str, etype: str = "proxy_error", headers: dict | None = None):
    return JSONResponse(
        status_code=status, 
        headers=headers,
        content={"detail": [{"loc": [], "msg": msg, "type": etype}]}
        )

def error(status: int, msg: str, etype: str = "proxy_error"):
    console.log(Text.assemble((f"ERROR {status} ", "bold red"), (msg, "red")))
    return error_response(status, msg, etype)

def upstream_message(payload) -> str:
    errs = payload.get("errors") if isinstance(payload, dict) else None
    if errs:
        return "; ".join(f"{e.get('code', '')} {e.get('message', '')}".strip() for e in errs)
    return json.dumps(payload)[:300]

class SyncError(Exception):
    pass

# --------------------------------------------------------------------------
# 3. Saving Usage
# --------------------------------------------------------------------------
class UsageStore:
    """Every day in one JSON file: {"version": 1, "days": {"2026-01-01": {...}, ...}}."""

    def __init__(self, path: str):
        self.path = path
        self.days : dict[str, dict] = {}
        self._last_save = 0.0
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as file:
                data = json.load(file)
            for key, day in data["days"].items():
                self.days[key] = {**new_day(), **day}  # tolerate files with missing fields
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as e:
            self.days = {}
            console.log(
                Text(f"Usage file unreadable ({type(e).__name__}): {e}", style="yellow")
            )

    def today(self) -> dict:
        return self.days.setdefault(utc_today(), new_day()) # assign if the value not exist

    def save(self, force: bool = False):
        now = time.monotonic()
        if not force and now - self._last_save < 1.0:
            return
        self._last_save = now
        tmp = self.path + ".tmp"
        try:  # write-then-rename so a crash can't leave a half-written file
            with open(tmp, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "version": 1,
                        "days": dict(sorted(self.days.items())),
                    },
                    file,
                    indent=2
                )
            os.replace(tmp, self.path)
        except OSError as e:
            console.log(Text(f"Could not save usage file {self.path}: {e}", style="bold red"))

    def record(self, ok: bool, questions: int, tokens_in: int, tokens_out: int, api_ms: int) -> None:
        day = self.today()
        day["requests"] += 1
        day["errors"] += 0 if ok else 1
        day["questions"] += questions
        day["input_tokens"] += tokens_in
        day["output_tokens"] += tokens_out
        day["api_ms_total"] += api_ms
        day["api_ms_max"] = max(day["api_ms_max"], api_ms)
        if MODEL not in day["models_used"]:
            day["models_used"].append(MODEL)
        self.save()

STORE = UsageStore(USAGE_FILE)

class Runtime:
    """Live counters that are not worth saving."""

    def __init__(self):
        self.in_flight = 0
        self.queued = 0
        self.completed = 0          
        self.read_marker = 0        
        self.reading = False
        self.last_error: str | None = None
        self.failures = 0
        self.dirty: asyncio.Event | None = None   
        self.settle = None                       

RT = Runtime()

# --------------------------------------------------------------------------
# 4. Reading neurons from Cloudflare (GraphQL Analytics)
# --------------------------------------------------------------------------

async def fetch_cloudflare_usage(client: httpx.AsyncClient, day: str) -> tuple[float, dict[str, float]]:
    """Return (account_total_neurons, neurons_by_model) for a UTC day."""
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end   = min(start + timedelta(days=1), datetime.now(timezone.utc) + timedelta(minutes=1))
    fmt   = "%Y-%m-%dT%H:%M:%SZ"

    query = """
    {
      viewer {
        accounts(filter: {accountTag: "%s"}) {
          aiInferenceAdaptiveGroups(
            limit: 1000,
            filter: {datetime_geq: "%s", datetime_lt: "%s"}
          ) {
            dimensions { modelId }
            sum { totalNeurons }
          }
        }
      }
    }
    """ % (ACCOUNT_ID, start.strftime(fmt), end.strftime(fmt))

    resp = await client.post(
        GRAPHQL_URL,
        json={"query": query},
        headers={"Authorization": f"Bearer {ANALYTICS_TOKEN}", "Cache-Control": "no-cache"} # blocking the caching for freach request
    )

    try:
        payload = resp.json()
    except json.JSONDecodeError:
        raise SyncError(f"HTTP {resp.status_code}: replay was not JSON")
    if not isinstance(payload, dict):
        raise SyncError(f"HTTP {resp.status_code}: unexpected replay")

    errs = payload.get("errors")
    if resp.status_code != 200 or errs:
        msg = ";".join(str(e.get("message", e)) if isinstance(e, dict) else str(e) for e in (errs or []))
        raise SyncError(msg or f"HTTP {resp.status_code}")
    accounts = ((payload.get("data") or {}).get("viewer") or {}).get("accounts") or []
    if not accounts:
        raise SyncError(
            "no account returned - check CLOUDFLARE_ACCOUNT_ID and " \
            "that the token has 'Account Analytics: Read'"
            )
    total, by_model = 0.0, {}
    # Neurons Total used by all models (clef and clef-flash)
    for g in accounts[0].get("aiInferenceAdaptiveGroups") or []:
        n = float((g.get("sum") or {}).get("totalNeurons") or 0)
        model_id = str((g.get("dimensions") or {}).get("modelId") or "unknown")
        total += n
        by_model[model_id] = by_model.get(model_id, 0.0) + n

    return total, by_model

async def refresh_once(app: FastAPI):
    d = utc_today()
    RT.reading, marker = True, RT.completed

    try:
        total, by_model = await fetch_cloudflare_usage(app.state.analytics_client, d)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        RT.last_error = f"{type(e).__name__}: {e}"
        RT.failures += 1
        if RT.failures == 1 or RT.failures % 10 == 0:
            console.log(
                Text.assemble(
                    ("USAGE ", "bold yellow"),
                    (f"could not read Cloudflare usage ({RT.last_error}); keeping the last reading", "yellow")
                )
            )
        return
    finally:
        RT.reading = False

    day                           = STORE.days.setdefault(d, new_day())
    day["cloudflare_neurons"]     = total
    day["cloudflare_by_model"]    = by_model
    day["neurons_read_at"]        = datetime.now(timezone.utc).isoformat(timespec="seconds")

    RT.read_marker, RT.last_error = marker, None
    if RT.failures:
        console.log(Text("USAGE read recovered", style="bold green"))
    RT.failures = 0
    STORE.save(force=True)
    console.log(
        Text.assemble(
            ("USAGE ", "bold magenta"),
            (f"cloudflare={total:,.1f} neurons today", "magenta"),
            (f"  {by_model}", "dim")
        )
    )

async def refresh_loop(app: FastAPI) -> None:
    """Wakes up when a request finished; reads at most once per USAGE_MIN_GAP seconds."""
    last = 0.0
    while True:
        await RT.dirty.wait()
        wait = USAGE_MIN_GAP - (time.monotonic() - last)
        if wait > 0:
            await asyncio.sleep(wait)
        RT.dirty.clear()
        await refresh_once(app)
        last = time.monotonic()

def _follow_up(step: int) -> None:
    RT.dirty.set()
    if step + 1 < len(USAGE_FOLLOW_UP):
        gap = USAGE_FOLLOW_UP[step + 1] - USAGE_FOLLOW_UP[step]
        RT.settle = asyncio.get_running_loop().call_later(gap, _follow_up, step + 1)

def request_refresh() -> None:
    """Call after every forwarded request: read now (coalesced), then again at USAGE_FOLLOW_UP
    seconds after the LAST request (each new request restarts that schedule)."""
    RT.completed += 1
    RT.dirty.set()
    if RT.settle:
        RT.settle.cancel()
    if USAGE_FOLLOW_UP:
        RT.settle = asyncio.get_running_loop().call_later(USAGE_FOLLOW_UP[0], _follow_up, 0)

# --------------------------------------------------------------------------
# 5. What we show: one log line per request + a live panel
# --------------------------------------------------------------------------
def log_request(ok: bool, info: dict, status: int, tokens_in: int = 0, tokens_out: int = 0,
                api_ms: int | None = None, queue_ms: int = 0, patched: int = 0, note: str = "") -> None:
    t = Text()
    t.append("OK  " if ok else f"ERR {status} ", style="bold green" if ok else "bold red")
    t.append(info["kinds"], style="bold")
    t.append(f'  "{info["text"]}"', style="italic")
    if info["rows"] is not None:
        t.append(f"  rows={info['rows']}", style="dim")
    if ok:
        t.append(f"  in={tokens_in:,} tok ({tokens_in / info['n']:,.0f}/q)  out={tokens_out:,}")
    if api_ms is not None:
        t.append(f"  api {api_ms} ms", style="cyan")
    if queue_ms >= 5:
        t.append(f"  queued {queue_ms} ms", style="yellow")
    if info["state_tokens"] >= STATE_TOKEN_WARN:
        t.append(f"  ⚠ state≈{info['state_tokens']:,} tok - Workers AI reportedly reads only the first ~2,048",
                 style="bold yellow")
    if patched:
        t.append(f"  (added instructions to {patched})", style="dim")
    if note:
        t.append(f"  {note}", style="red")
    console.log(t)

def describe(body: dict) -> dict:
    """What a request is about (call BEFORE the body is modified)."""
    questions = {k: q for k, q in body["questions"].items() if isinstance(q, dict)}
    kinds = Counter(str(q.get("type", "?")) for q in questions.values())
    state = body.get("state")
    text, rows = "", None

    if isinstance(state, dict):
        text = str(state.get("condition") or "")
        if isinstance(state.get("rows"), list):
            rows = len(state["rows"])
    if not text:
        first = next((q["instructions"] for q in questions.values() if q.get("instructions")), "")
        text = str(first) if first else (state if isinstance(state, str) else "")
    text = " ".join(str(text).split())

    if len(text) > 70:
        text = text[:69] + "..."
    state_chars = len(json.dumps(state, ensure_ascii=False)) if state is not None else 0
    return {
        "kinds": ", ".join(f"{k}×{n}" for k, n in kinds.items()) or "no questions",
        "text": text,
        "rows": rows,
        "n": max(1, len(body["questions"])),
        "state_tokens": state_chars // 4,   # rough: ~4 characters per token
    }
# --------------------------------------------------------------------------
# 6. Routes
# --------------------------------------------------------------------------
@app.get("/health")
async def health():
    return JSONResponse(
        {"status": "ok", "model": MODEL, "usage": 0}
    )

@app.get("/v1/model")
async def models():
    model = MODELS.get(MODEL)

    if model is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Model Not Found",
                "requested_model": MODEL,
                "available_models": list(MODELS.keys())
            }
        )
    
    return JSONResponse(
        {"model": model}
    )

@app.post("/{path:path}")
async def proxy(path: str, request: Request):
    t0 = time.perf_counter()
    # Verified request
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return error(400, "request body is not valid JSON")
    if not isinstance(body, dict) or "questions" not in body:
        return error(422, "body must contain 'state' and 'questions")
    info = describe(body)
    # (a) Whatever model the extension asks for (e.g. jev-latest), use ours.
    body["model"] = MODEL

    # (b) Workers AI requires `instructions` on every question.
    patched = 0
    for q in (body.get("questions") or {}).values():
        if isinstance(q, dict) and not q.get("instructions"):
            q["instructions"] = FALLBACK_INSTRUCTIONS
            patched += 1
    # (c) Budget guard: Cloudflare's last reading for today
    today = STORE.today()
    used = today["cloudflare_neurons"]
    if used is not None and used >= DAILY_NEURON_LIMIT:
        today["blocked"] += 1
        STORE.save()
        msg = (f"daily neuron budget reached ({DAILY_NEURON_LIMIT:g}; Cloudflare reported {used:,.0f}); "
               "raise DAILY_NEURON_LIMIT or wait for the UTC day to roll over")
        log_request(False, info, 403, note=msg)
        return error_response(403, msg, "budget_exceeded")
    # (d) Forward. The semaphore makes excess requests WAIT here instead of hammering Cloudflare
    t_wait = time.perf_counter()
    RT.queued += 1
    try:
        await request.app.state.limiter.acquire()
    finally:
        RT.queued -= 1
    queue_ms = int((time.perf_counter() - t_wait) * 1000)

    RT.in_flight += 1
    t_api = time.perf_counter()
    response, failure, headers = None, None, None
    try:
        response = await request.app.state.client.post(UPSTREAM_URL, json=body)
    except httpx.TimeoutException:
        failure = (504, "Workers AI timed out", "upstream_timeout")
    except httpx.HTTPError as e:
        failure = (502, f"could not reach Workers AI: {type(e).__name__}: {e}", "upstream_error")
    finally:
        RT.in_flight -= 1
        request.app.state.limiter.release()
    api_ms = int((time.perf_counter() - t_api) * 1000)

    result = None
    if failure is None:
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = {"raw": response.text[:300]}
        if response.status_code == 200 and isinstance(payload, dict) and payload.get("success", True):
            result = payload.get("result", payload)
        else:
            retry_after = response.headers.get("retry-after")
            headers = {"Retry-After": retry_after} if retry_after else None
            failure = (response.status_code if response.status_code != 200 else 502,
                       "Workers AI: " + upstream_message(payload), "upstream_error")
    if failure:
        STORE.record(False, info["n"], 0, 0, api_ms)
        request_refresh()
        log_request(False, info, failure[0], api_ms=api_ms, queue_ms=queue_ms, note=failure[1])
        return error_response(failure[0], failure[1], failure[2], headers)
    
    # (e) Unwrap {"result": {...}} into the plain Jev/SystemOne response.
    usage_info = result.get("usage") or {}
    tokens_in, tokens_out = int(usage_info.get("input_tokens", 0)), int(usage_info.get("output_tokens", 0))
    STORE.record(True, info["n"], tokens_in, tokens_out, api_ms)
    request_refresh()
    log_request(True, info, 200, tokens_in, tokens_out, api_ms, queue_ms, patched)
    return JSONResponse(result)

# --------------------------------------------------------------------------
# 7. Entry point
# --------------------------------------------------------------------------
def banner() -> None:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="cyan", justify="right")
    t.add_column()
    t.add_row("listen", f"http://{HOST}:{PORT}")
    t.add_row("upstream", f"@cf/cloudflare/{MODEL}")
    t.add_row("concurrency", f"max {MAX_CONCURRENCY} in flight")
    t.add_row("budget guard", f"{DAILY_NEURON_LIMIT:,.0f} neurons per UTC day (Cloudflare's figure)")
    t.add_row("usage read", f"after requests (every {USAGE_MIN_GAP:g}s at most), then {USAGE_FOLLOW_UP_TEXT}s after the last")
    t.add_row("history", USAGE_FILE)
    console.print(Panel(t, title="[bold]clef-proxy[/]", border_style="cyan", expand=False))
def main() -> None:
    if MODEL not in MODELS:
        sys.exit(f"CLEF_MODEL must be one of {sorted(MODELS)}, got {MODEL!r}")
    if not ACCOUNT_ID or not API_TOKEN:
        sys.exit("Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN (put them in your .env file)")
    banner()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning", log_config=None)
    
if __name__ == "__main__":
    main()