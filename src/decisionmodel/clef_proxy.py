import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
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
ANALYTICS_LAG = float(os.getenv("ANALYTICS_LAG", "120"))
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
    yield

    await app.state.client.aclose()
    await app.state.analytics_client.aclose()

app = FastAPI(title="clef-proxy", lifespan=lifespan)

# --------------------------------------------------------------------------
# 2. Helpers
# --------------------------------------------------------------------------
def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%M-%d")

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

def error(status: int, msg: str, etype: str = "proxy_error", headers: dict | None = None):
    console.log(Text.assemble((f"ERROR {status} ", "bold red"), (msg, "red")))
    return JSONResponse(
        status_code=status,
        headers=headers,
        content={"detail": [{"loc": [], "msg": msg, "type": etype}]}
        )

def upstream_message(payload) -> str:
    errs = payload.get("errors") if isinstance(payload, dict) else None
    if errs:
        return "; ".join(f"{e.get('code', '')} {e.get('message', '')}".strip() for e in errs)
    return json.dumps(payload)[:300]

class SyncError(Exception):
    pass
# --------------------------------------------------------------------------
# 3. Reading neurons from Cloudflare (GraphQL Analytics)
# --------------------------------------------------------------------------

async def fetch_cloudflare_usage(client: httpx.AsyncClient, day: str) -> tuple[float, dict[str, float]]:
    """Return (account_total_neurons, neurons_by_model) for a UTC day."""
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end   = start + timedelta(days=1)
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
        headers={"Authorization": f"Bearer {ANALYTICS_TOKEN}"}
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
    for g in accounts[0].get("aiInferenceAdaptiveGroups"):
        n = float((g.get("sum") or {}).get("totalNeurons") or 0)
        model_id = str((g.get("dimensions") or {}).get("modelId") or "unknown")
        total += n
        by_model[model_id] = by_model.get(model_id, 0.0) + n

    return total, by_model


# --------------------------------------------------------------------------
# 4. Routes
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
    
    # (a) Whatever model the extension asks for (e.g. jev-latest), use ours.
    body["model"] = MODEL

        # (b) Workers AI requires `instructions` on every question.
    patched = 0
    for q in (body.get("questions") or {}).values():
        if isinstance(q, dict) and not q.get("instructions"):
            q["instructions"] = FALLBACK_INSTRUCTIONS
            patched += 1
    # (d) Forward. The semaphore makes excess requests WAIT here instead of hammering Cloudflare
    client: httpx.AsyncClient = request.app.state.client
    try:
        async with request.app.state.limiter:
            response = await client.post(UPSTREAM_URL, json=body)
    except httpx.TimeoutException:
        return error(504, "Workers AI timed out", "upstream_timeout")
    except httpx.HTTPError as e:
        return error(502, f"Could not reach Workers AI: {type(e).__name__}: {e}", "upstream_error")

    try:
        payload = response.json()
    except json.JSONDecodeError:
        payload = {"raw": response.text[:300]}

    # print(payload)
    ok = response.status_code == 200 and isinstance(payload, dict) and payload.get("success", True)

    if not ok:
        # Pass the status through (429/5xx are retried by the extension's own
        retry_after = response.headers.get("retry-after")
        return error(
            response.status_code if response.status_code != 200 else 502,
            "Workers AI: " + upstream_message(payload), "upstream_error",
            headers={"Retry-After": retry_after} if retry_after else None
        )

    # (e) Unwrap {"result": {...}} into the plain Jev/SystemOne response.
    result = payload.get("result", payload)
    return JSONResponse(result)

def main() -> None:
    if MODEL not in MODELS:
        sys.exit(f"CLEF_MODEL must be one of {sorted(MODELS)}, got {MODEL!r}")
    if not ACCOUNT_ID or not API_TOKEN:
        sys.exit("Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN (put them in your .env file)")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning", log_config=None)
    
if __name__ == "__main__":
    main()