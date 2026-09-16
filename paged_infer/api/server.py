"""FastAPI production server with OpenAI-compatible streaming endpoints and real-time Telemetry."""

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from paged_infer.engine import AsyncPagedInferEngine
from paged_infer.models.transformer import ModelConfig
from paged_infer.scheduler.sequence import SamplingParams

DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"


class CompletionRequest(BaseModel):
    model: str = "paged-infer-v1"
    prompt: str
    max_tokens: int = Field(default=64, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=-1)
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "paged-infer-v1"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=64, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False


class BurstRequest(BaseModel):
    count: int = Field(default=5, ge=1, le=30)
    max_tokens: int = Field(default=32, ge=5, le=128)


def create_app(engine: Optional[AsyncPagedInferEngine] = None) -> FastAPI:
    """Factory creating and configuring the FastAPI server application."""
    active_engine = engine or AsyncPagedInferEngine(
        model_config=ModelConfig(vocab_size=4096, hidden_size=256, num_layers=4, num_heads=8, block_size=16),
        num_blocks=96,
        block_size=16,
        max_batch_size=16,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await active_engine.start()
        yield
        await active_engine.stop()

    app = FastAPI(
        title="PagedInfer Serving Engine",
        description="High-Throughput LLM Serving with PagedAttention and Continuous Batching",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.engine = active_engine

    @app.get("/healthz")
    async def healthcheck():
        return {"status": "healthy", "service": "paged-infer"}

    @app.get("/v1/engine/metrics")
    async def get_metrics():
        return active_engine.get_telemetry()

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    async def get_dashboard():
        if DASHBOARD_PATH.exists():
            return HTMLResponse(content=DASHBOARD_PATH.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>PagedInfer Dashboard</h1><p>Dashboard file loading...</p>")

    @app.post("/v1/completions")
    async def create_completion(req: CompletionRequest):
        params = SamplingParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
        )

        if req.stream:
            async def event_generator():
                async for token in active_engine.add_request(req.prompt, params):
                    chunk = {
                        "id": "cmpl-" + str(int(time.time() * 1000)),
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{"text": token, "index": 0, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")
        else:
            result = await active_engine.generate_full(req.prompt, params)
            return {
                "id": "cmpl-" + str(int(time.time() * 1000)),
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"text": result["text"], "index": 0, "finish_reason": "stop"}],
                "usage": {"completion_tokens": result["num_tokens"]},
            }

    @app.post("/v1/chat/completions")
    async def create_chat_completion(req: ChatCompletionRequest):
        prompt = "\n".join([f"{m.role}: {m.content}" for m in req.messages]) + "\nassistant:"
        params = SamplingParams(max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p)

        if req.stream:
            async def event_generator():
                async for token in active_engine.add_request(prompt, params):
                    chunk = {
                        "id": "chatcmpl-" + str(int(time.time() * 1000)),
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{"delta": {"content": token}, "index": 0, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")
        else:
            result = await active_engine.generate_full(prompt, params)
            return {
                "id": "chatcmpl-" + str(int(time.time() * 1000)),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "message": {"role": "assistant", "content": result["text"]},
                    "index": 0,
                    "finish_reason": "stop",
                }],
                "usage": {"completion_tokens": result["num_tokens"]},
            }

    @app.post("/v1/demo/burst")
    async def trigger_burst(burst: BurstRequest):
        """Simulates concurrent client requests for live HUD demonstration."""
        sample_prompts = [
            "Explain virtual memory paging in operating systems",
            "What is continuous batching in modern LLM serving?",
            "Compare paged attention with naive contiguous tensors",
            "How does copy on write reduce prompt duplicate storage?",
            "Describe the trade-off between throughput and TTFT latency",
        ]

        async def fire(i: int):
            prompt = sample_prompts[i % len(sample_prompts)]
            params = SamplingParams(max_tokens=burst.max_tokens, temperature=0.7)
            tokens = []
            async for chunk in active_engine.add_request(prompt, params):
                tokens.append(chunk)
            return len(tokens)

        # Fire concurrently in background
        for i in range(burst.count):
            asyncio.create_task(fire(i))

        return {"status": "burst_scheduled", "count": burst.count}

    return app


app = create_app()


def main():
    """CLI entrypoint for running the serving engine server."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="PagedInfer Serving Engine Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    print(f"🚀 Starting PagedInfer Serving Engine on http://{args.host}:{args.port}")
    print(f"📊 Real-time Visual Dashboard: http://localhost:{args.port}/dashboard")
    uvicorn.run("paged_infer.api.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
