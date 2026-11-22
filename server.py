"""FastAPI Streaming Backend for GPT-98.4M Multi-Turn Assistant Dashboard.

Features:
- Live local PyTorch inference running directly on CUDA (RTX 4050 6GB)
- Multi-model registry supporting DPO, SFT, and Pretrain checkpoints
- Interactive Chat completions with context tracking and Rule 7 stop criteria
- Real-time token streaming via Server-Sent Events (SSE) and fast JSON responses
- Serves frontend/dashboard.html and frontend/asset/bg.png
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import sys
import time
from typing import AsyncGenerator, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

import config
from model.transformer import GPT, GPTConfig

BOS_ID = 1
EOS_ID = 2

STOP_STRINGS = [
    "\n### User:",
    "### User:",
    "\n### Assistant:",
    "### Assistant:",
    "\n###",
]

MODEL_REGISTRY = {
    "Final Model (DPO)": PROJECT_ROOT / "experiments/20261122_014711_dpo_lr1e6_b01_20261122_014642/checkpoints/best.pt",
    "SFT Checkpoint (v2)": PROJECT_ROOT / "experiments/20261121_174835_gpt_multi_turn_sft/checkpoints/best.pt",
    "Continued Pretrain": PROJECT_ROOT / "experiments/20261121_023140_gpt_continued_pretrain/checkpoints/best.pt",
    "Base Pretrain (600M)": PROJECT_ROOT / "experiments/20261104_211024_gpt_pretrain/checkpoints/best.pt",
}

DEFAULT_MODEL_NAME = "Final Model (DPO)"


class ModelManager:
    """Thread-safe manager for loading and serving GPT checkpoints on CUDA."""

    def __init__(self, device: str = config.DEVICE):
        self.device = device
        self.tokenizer: Optional[Tokenizer] = None
        self.model: Optional[GPT] = None
        self.active_model_name: Optional[str] = None
        self.model_config: Optional[GPTConfig] = None
        self.lock = asyncio.Lock()

    def init_tokenizer(self):
        tok_file = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
        if not tok_file.exists():
            tok_file = config.TOKENIZER_FILE
        print(f"[Backend] Loading tokenizer from: {tok_file}")
        self.tokenizer = Tokenizer.from_file(str(tok_file))

    def load_checkpoint(self, model_name: str) -> bool:
        if self.tokenizer is None:
            self.init_tokenizer()

        if model_name not in MODEL_REGISTRY:
            # Fallback to closest match
            matched = None
            for k in MODEL_REGISTRY:
                if model_name.lower() in k.lower() or k.lower() in model_name.lower():
                    matched = k
                    break
            if matched:
                model_name = matched
            else:
                model_name = DEFAULT_MODEL_NAME

        if self.active_model_name == model_name and self.model is not None:
            return True

        ckpt_path = MODEL_REGISTRY[model_name]
        if not ckpt_path.exists():
            print(f"[Backend Warning] Checkpoint {ckpt_path} not found. Falling back to default.")
            ckpt_path = MODEL_REGISTRY[DEFAULT_MODEL_NAME]
            model_name = DEFAULT_MODEL_NAME

        print(f"[Backend] Loading model checkpoint: {model_name} -> {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)

        saved_cfg = ckpt.get("config_snapshot", {})
        block_size = saved_cfg.get("block_size", config.BLOCK_SIZE)

        gpt_config = GPTConfig.from_tokenizer_config(
            config_path=config.TOKENIZER_CONFIG_FILE,
            block_size=block_size,
            n_layer=saved_cfg.get("n_layer", config.N_LAYER),
            n_head=saved_cfg.get("n_head", config.N_HEAD),
            n_embd=saved_cfg.get("n_embd", config.N_EMBD),
            ffn_dim=saved_cfg.get("ffn_dim", config.FFN_DIM),
            dropout=0.0,
            bias=saved_cfg.get("bias", config.BIAS),
            tie_weights=saved_cfg.get("tie_weights", config.TIE_WEIGHTS),
            use_flash=True,
        )

        model = GPT(gpt_config).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        self.model = model
        self.model_config = gpt_config
        self.active_model_name = model_name
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        print(f"[Backend] {model_name} ready on {self.device.upper()} (VRAM: {vram_mb:.1f} MB)\n")
        return True

    def format_tokens(self, history: list[dict[str, str]], new_message: str, max_context_len: int = 1024) -> list[int]:
        assert self.tokenizer is not None
        input_ids = [BOS_ID]

        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if not content:
                continue
            if role == "user":
                turn_ids = self.tokenizer.encode("### User:\n").ids + self.tokenizer.encode(f"{content}\n\n").ids
                input_ids.extend(turn_ids)
            elif role == "assistant":
                turn_ids = (
                    self.tokenizer.encode("### Assistant:\n").ids
                    + self.tokenizer.encode(content).ids
                    + [EOS_ID]
                    + self.tokenizer.encode("\n\n").ids
                )
                input_ids.extend(turn_ids)

        cur_user_ids = self.tokenizer.encode("### User:\n").ids + self.tokenizer.encode(f"{new_message.strip()}\n\n").ids
        asst_header_ids = self.tokenizer.encode("### Assistant:\n").ids
        input_ids.extend(cur_user_ids + asst_header_ids)

        max_prompt_len = max_context_len - 150
        if len(input_ids) > max_prompt_len:
            input_ids = [BOS_ID] + input_ids[-(max_prompt_len - 1):]

        return input_ids

    def clean_response(self, text: str) -> str:
        clean = text.strip()
        for stop_str in STOP_STRINGS:
            pos = clean.find(stop_str)
            if pos != -1:
                clean = clean[:pos].strip()
        return clean

    @torch.no_grad()
    def generate_full(
        self,
        prompt: str,
        history: list[dict[str, str]],
        temperature: float = 0.4,
        top_p: float = 0.90,
        min_p: float = 0.08,
        repetition_penalty: float = 1.15,
        max_new_tokens: int = 200,
    ) -> tuple[str, float, int]:
        assert self.model is not None and self.tokenizer is not None
        token_ids = self.format_tokens(history=history, new_message=prompt)
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        t0 = time.perf_counter()
        out_ids = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=4,
            eos_token_id=EOS_ID,
        )
        elapsed = time.perf_counter() - t0

        gen_tokens = out_ids[0, len(token_ids):].tolist()
        if EOS_ID in gen_tokens:
            gen_tokens = gen_tokens[: gen_tokens.index(EOS_ID)]

        raw_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        clean_text = self.clean_response(raw_text)
        speed = len(gen_tokens) / max(elapsed, 1e-6)

        return clean_text, round(speed, 1), len(gen_tokens)

    @torch.no_grad()
    async def generate_stream(
        self,
        prompt: str,
        history: list[dict[str, str]],
        temperature: float = 0.4,
        top_p: float = 0.90,
        min_p: float = 0.08,
        repetition_penalty: float = 1.15,
        max_new_tokens: int = 200,
    ) -> AsyncGenerator[dict, None]:
        assert self.model is not None and self.tokenizer is not None
        token_ids = self.format_tokens(history=history, new_message=prompt)
        cur_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        t0 = time.perf_counter()
        generated_token_ids: list[int] = []
        accumulated_text = ""

        for step in range(max_new_tokens):
            # Crop to block_size if needed
            idx_cond = cur_ids if cur_ids.size(1) <= config.BLOCK_SIZE else cur_ids[:, -config.BLOCK_SIZE:]
            logits = self.model(idx_cond)
            logits = logits[:, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0 and generated_token_ids:
                seen = set(generated_token_ids)
                for tid in seen:
                    if logits[0, tid] > 0:
                        logits[0, tid] /= repetition_penalty
                    else:
                        logits[0, tid] *= repetition_penalty

            # Temperature
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)

                # Min-p
                if min_p is not None and min_p > 0.0:
                    max_prob = torch.max(probs, dim=-1, keepdim=True).values
                    thresh = min_p * max_prob
                    probs[probs < thresh] = 0.0
                    probs = probs / torch.sum(probs, dim=-1, keepdim=True)

                # Top-p
                if top_p is not None and top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum_probs - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=True)
                    probs.scatter_(1, sorted_indices, sorted_probs)

                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            tid = next_token.item()
            cur_ids = torch.cat([cur_ids, next_token], dim=1)
            generated_token_ids.append(tid)

            # Check EOS
            if tid == EOS_ID:
                break

            # Decode incremental token
            new_text = self.tokenizer.decode([tid], skip_special_tokens=True)
            accumulated_text += new_text

            # Check stop strings
            hit_stop = False
            for stop_str in STOP_STRINGS:
                if stop_str in accumulated_text:
                    hit_stop = True
                    break
            if hit_stop:
                break

            yield {"token": new_text, "done": False}
            await asyncio.sleep(0.001)

        elapsed = time.perf_counter() - t0
        speed = len(generated_token_ids) / max(elapsed, 1e-6)
        yield {"token": "", "done": True, "speed": round(speed, 1), "tokens": len(generated_token_ids)}


# Global Model Manager
manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize tokenizer and default model
    manager.init_tokenizer()
    manager.load_checkpoint(DEFAULT_MODEL_NAME)
    yield


app = FastAPI(
    title="GPT-98.4M Multi-Turn Assistant Backend",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local development & browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Schemas
class ChatRequest(BaseModel):
    prompt: str
    history: list[dict[str, str]] = Field(default_factory=list)
    model: str = Field(default="Final Model (DPO)")
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    top_p: float = Field(default=0.90, ge=0.0, le=1.0)
    min_p: float = Field(default=0.08, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.15, ge=1.0, le=3.0)
    max_new_tokens: int = Field(default=200, ge=1, le=512)
    stream: bool = Field(default=False)


class ModelSwitchRequest(BaseModel):
    model: str


# Endpoints
@app.get("/")
@app.get("/dashboard.html")
async def root():
    """Serve the primary dashboard interface."""
    index_file = PROJECT_ROOT / "frontend" / "dashboard.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "GPT-98.4M Backend is live. Open /frontend/dashboard.html"}


@app.get("/v1/models")
async def get_models():
    """List available checkpoints and currently active model."""
    return {
        "active_model": manager.active_model_name,
        "models": list(MODEL_REGISTRY.keys()),
    }


@app.post("/v1/models/switch")
async def switch_model(req: ModelSwitchRequest):
    """Switch active model checkpoint."""
    async with manager.lock:
        success = manager.load_checkpoint(req.model)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to load model")
        return {"active_model": manager.active_model_name, "status": "loaded"}


@app.get("/v1/system")
async def get_system_telemetry():
    """Retrieve GPU and model hardware telemetry."""
    vram_alloc = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
    vram_res = torch.cuda.memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    return {
        "device_name": device_name,
        "vram_allocated_mb": round(vram_alloc, 1),
        "vram_reserved_mb": round(vram_res, 1),
        "total_vram_mb": 6144,
        "active_model": manager.active_model_name,
        "context_window": config.BLOCK_SIZE,
        "vocab_size": config.VOCAB_SIZE,
        "param_count": 98425344,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    }


@app.post("/v1/chat")
async def chat_completion(req: ChatRequest):
    """Chat completion endpoint supporting both JSON and SSE streaming."""
    async with manager.lock:
        if req.model and req.model != manager.active_model_name:
            manager.load_checkpoint(req.model)

        if req.stream:
            async def event_generator():
                async for chunk in manager.generate_stream(
                    prompt=req.prompt,
                    history=req.history,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    min_p=req.min_p,
                    repetition_penalty=req.repetition_penalty,
                    max_new_tokens=req.max_new_tokens,
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        reply, speed, tokens = manager.generate_full(
            prompt=req.prompt,
            history=req.history,
            temperature=req.temperature,
            top_p=req.top_p,
            min_p=req.min_p,
            repetition_penalty=req.repetition_penalty,
            max_new_tokens=req.max_new_tokens,
        )

        return {
            "response": reply,
            "text": reply,
            "speed_tok_s": speed,
            "tokens": tokens,
            "model": manager.active_model_name,
        }


# Mount static assets
frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")
    asset_dir = frontend_dir / "asset"
    if asset_dir.exists():
        app.mount("/asset", StaticFiles(directory=str(asset_dir)), name="asset")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("  GPT-98.4M FASTAPI SERVER STARTING")
    print("  Dashboard UI : http://localhost:8000")
    print("  API Endpoint : http://localhost:8000/v1/chat")
    print("  Telemetry    : http://localhost:8000/v1/system")
    print("=" * 70 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
