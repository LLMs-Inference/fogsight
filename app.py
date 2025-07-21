import os
import json
import asyncio
from datetime import datetime
from typing import AsyncGenerator, List, Optional

import pytz
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types
from openai import AsyncOpenAI, OpenAIError


# -----------------------------------------------------------------------
# 0. 配置
# -----------------------------------------------------------------------
shanghai_tz = pytz.timezone("Asia/Shanghai")

credentials = json.load(open("credentials.json"))
API_KEY = credentials["API_KEY"]
BASE_URL = credentials.get("BASE_URL", "")

if API_KEY.startswith("sk-REPLACE_ME"):
    raise RuntimeError("请在环境变量里配置 API_KEY")

USE_GEMINI = True
client = None

if USE_GEMINI:
    gemini_client = genai.Client(api_key=API_KEY, http_options=types.HttpOptions(base_url=BASE_URL))
else:
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

templates = Jinja2Templates(directory="templates")

# -----------------------------------------------------------------------
# 1. FastAPI 初始化
# -----------------------------------------------------------------------
app = FastAPI(title="AI Animation Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    topic: str
    history: Optional[List[dict]] = None


# -----------------------------------------------------------------------
# 2. 核心：流式生成器 (现在会使用 history)
# -----------------------------------------------------------------------
async def llm_event_stream(
    topic: str,
    history: Optional[List[dict]] = None,
    model: str = "gemini-2.5-pro",  # Changed model for better performance if needed
) -> AsyncGenerator[str, None]:
    history = history or []

    # The system prompt is now more focused
    system_prompt = f"""
你的任务是创建出一个极为精美的动态动画，该动画要详细阐释 {topic}。动画需具备动态效果，要如同一个正在播放的完整视频，包含一个完整的流程，以便能够清晰地讲解相关知识点。页面设计要极为精美、美观且富有设计感，同时要能够出色地传达知识和逻辑，确保知识内容和图像展示准确无误。此外，动画要附带一些旁白式的文字解说，从头到尾清晰地讲解一个小知识点。动画不需要设置任何互动按钮，直接开始播放即可。动画应采用和谐好看且广泛被采用的浅色配色方案，并运用大量丰富的视觉元素。动画要配备中英双语字幕。请务必保证任何一个元素都在一个 2K 分辨率的容器中被正确摆放，避免出现穿模、字幕遮挡、图形位置错误等影响正确视觉传达的问题。整个项目需使用 HTML、CSS、JavaScript 和 SVG 技术实现，并将代码整合到一个 HTML 文件中。
""".strip()

    if USE_GEMINI:
        try:
            full_prompt = system_prompt + "\n\n" + topic
            if history:
                history_text = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in history]
                )
                full_prompt = history_text + "\n\n" + full_prompt

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: gemini_client.models.generate_content(
                    model="gemini-2.5-pro", contents=full_prompt
                ),
            )

            text = response.text
            chunk_size = 50

            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                payload = json.dumps({"token": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.05)

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": topic},
        ]

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.8,
            )
        except OpenAIError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        async for chunk in response:
            token = chunk.choices[0].delta.content or ""
            if token:
                payload = json.dumps({"token": token}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.001)

    yield 'data: {"event":"[DONE]"}\n\n'


# -----------------------------------------------------------------------
# 3. 路由 (CHANGED: Now a POST request)
# -----------------------------------------------------------------------
@app.post("/generate")
async def generate(
    chat_request: ChatRequest,  # CHANGED: Use the Pydantic model
    request: Request,
):
    """
    Main endpoint: POST /generate
    Accepts a JSON body with "topic" and optional "history".
    Returns an SSE stream.
    """
    accumulated_response = ""  # for caching flow results

    async def event_generator():
        nonlocal accumulated_response
        try:
            async for chunk in llm_event_stream(
                chat_request.topic, chat_request.history
            ):
                accumulated_response += chunk
                if await request.is_disconnected():
                    break
                yield chunk
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    async def wrapped_stream():
        async for chunk in event_generator():
            yield chunk

    headers = {
        "Cache-Control": "no-store",
        "Content-Type": "text/event-stream; charset=utf-8",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(wrapped_stream(), headers=headers)


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "time": datetime.now(shanghai_tz).strftime("%Y%m%d%H%M%S"),
        },
    )


# -----------------------------------------------------------------------
# 4. 本地启动命令
# -----------------------------------------------------------------------
# uvicorn app:app --reload --host 0.0.0.0 --port 8000


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
