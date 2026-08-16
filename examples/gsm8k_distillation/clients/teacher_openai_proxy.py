"""Minimal OpenAI-compatible chat-completions proxy in front of TeacherClient.

Lets NeMo Data Designer (which speaks OpenAI's chat-completions HTTP API) call the same
Bedrock-hosted Qwen3-32B teacher used by run.py, instead of a separate NVIDIA-hosted model.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from examples.gsm8k_distillation.clients.teacher_client import TeacherClient

app = FastAPI()
_teacher = TeacherClient()


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict]

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(part.get("text", "") for part in self.content)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.9
    max_tokens: int = 256


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> dict:
    prompt = next((m.text() for m in reversed(req.messages) if m.role == "user"), None)
    if prompt is None:
        raise HTTPException(status_code=400, detail="No message with role 'user' in request")
    [completion] = _teacher.generate([prompt], max_tokens=req.max_tokens, temperature=req.temperature)
    return {
        "id": "chatcmpl-teacher-proxy",
        "object": "chat.completion",
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": completion},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def serve(port: int) -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
