"""
OpenMythos inference server.

Endpoints:
  POST /generate   — text in, generated text out
  POST /forward    — text in, raw logits out (last token)
  GET  /health     — liveness check

Auth: Bearer token via API_KEY env var. Set to empty string to disable.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from open_mythos.main import MythosConfig, OpenMythos
from open_mythos.tokenizer import MythosTokenizer

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

API_KEY = os.getenv("API_KEY", "")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "")  # empty = random weights

# Model hyperparams (override via env for different configs)
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "200000"))
DIM = int(os.getenv("DIM", "256"))
N_HEADS = int(os.getenv("N_HEADS", "8"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "512"))
MAX_LOOP_ITERS = int(os.getenv("MAX_LOOP_ITERS", "4"))
ATTN_TYPE = os.getenv("ATTN_TYPE", "mla")

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = MythosConfig(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        n_heads=N_HEADS,
        max_seq_len=MAX_SEQ_LEN,
        max_loop_iters=MAX_LOOP_ITERS,
        prelude_layers=1,
        coda_layers=1,
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=64,
        lora_rank=8,
        attn_type=ATTN_TYPE,
        n_kv_heads=8,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
    )
    model = OpenMythos(cfg).to(DEVICE).eval()

    if CHECKPOINT_PATH and os.path.isfile(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        print("No checkpoint loaded — model running with random weights.")

    state["model"] = model
    state["tokenizer"] = MythosTokenizer()
    state["cfg"] = cfg
    print(f"Model ready on {DEVICE}")
    yield
    state.clear()


app = FastAPI(title="OpenMythos API", version="0.1.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

bearer = HTTPBearer(auto_error=False)


def verify_key(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
    if not API_KEY:
        return  # auth disabled
    if creds is None or creds.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(64, ge=1, le=512)
    n_loops: int = Field(8, ge=1, le=64)
    temperature: float = Field(1.0, gt=0.0, le=2.0)
    top_k: int = Field(50, ge=0, le=200)


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    generated_tokens: int


class ForwardRequest(BaseModel):
    prompt: str
    n_loops: int = Field(4, ge=1, le=64)


class ForwardResponse(BaseModel):
    logits_shape: list[int]
    top_token_ids: list[int]
    top_token_probs: list[float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "checkpoint": CHECKPOINT_PATH or "none (random weights)",
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, _=Depends(verify_key)):
    tok = state["tokenizer"]
    model: OpenMythos = state["model"]

    prompt_ids = tok.encode(req.prompt)
    if not prompt_ids:
        raise HTTPException(status_code=400, detail="Empty prompt after tokenization")

    input_ids = torch.tensor([prompt_ids], device=DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=req.max_new_tokens,
            n_loops=req.n_loops,
            temperature=req.temperature,
            top_k=req.top_k,
        )

    generated = output_ids[0, len(prompt_ids) :].tolist()
    return GenerateResponse(
        text=tok.decode(generated),
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated),
    )


@app.post("/forward", response_model=ForwardResponse)
def forward(req: ForwardRequest, _=Depends(verify_key)):
    tok = state["tokenizer"]
    model: OpenMythos = state["model"]

    prompt_ids = tok.encode(req.prompt)
    if not prompt_ids:
        raise HTTPException(status_code=400, detail="Empty prompt after tokenization")

    input_ids = torch.tensor([prompt_ids], device=DEVICE)

    with torch.no_grad():
        logits = model(input_ids, n_loops=req.n_loops)

    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    top_probs, top_ids = probs.topk(10)

    return ForwardResponse(
        logits_shape=list(logits.shape),
        top_token_ids=top_ids.tolist(),
        top_token_probs=top_probs.tolist(),
    )
