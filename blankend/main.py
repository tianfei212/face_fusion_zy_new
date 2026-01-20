from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
from .streaming.routes import get_router, start_relay
from .resources.routes import get_resources_router
from pathlib import Path
from .video_processing import get_video_processing_router
from .skills_api import get_skills_router
from .realtime_pipeline import get_realtime_pipeline_router
from .ai_core import get_ai_core_router
import json
import logging
def _load_config() -> dict:
    path = Path(__file__).resolve().parent / "config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

app = FastAPI()
logger = logging.getLogger("blankend")

@app.on_event("startup")
async def start_stream_relay() -> None:
    try:
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(level=logging.INFO)
        from .video_processing.gpu import log_gpu_summary

        log_gpu_summary("startup_gpu")
    except Exception:
        pass
    try:
        from .video_processing.codec import probe_hw_codecs

        st = probe_hw_codecs()
        nvidia = (st.get("nvidia") or {}) if isinstance(st, dict) else {}
        logging.getLogger("blankend.video_processing.codec").info(
            "startup_codec ffmpeg=%s cuda_hwaccel=%s h264_nvenc=%s h264_cuvid=%s",
            (st.get("ffmpeg") or {}).get("installed") if isinstance(st, dict) else None,
            nvidia.get("cuda_hwaccel"),
            nvidia.get("h264_nvenc"),
            nvidia.get("h264_cuvid"),
        )
    except Exception:
        pass
    start_relay(asyncio.get_running_loop())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def cache_static_files(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/files/"):
        config = _load_config()
        cache_config = config.get("cache", {}) if isinstance(config, dict) else {}
        max_age = cache_config.get("static_max_age", 31536000)
        response.headers.setdefault("Cache-Control", f"public, max-age={max_age}, immutable")
    return response

@app.get("/")
async def root():
    return {"status": "ok"}

@app.head("/")
async def head_root():
    return {"status": "ok"}

app.include_router(get_router())
app.include_router(get_resources_router())
app.include_router(get_video_processing_router())
app.include_router(get_skills_router())
app.include_router(get_realtime_pipeline_router())
app.include_router(get_ai_core_router())

# Static mounts for assets to serve files returned by resource routes
_assets_base = Path(__file__).resolve().parent / 'assets'
app.mount('/files/human_pic', StaticFiles(directory=_assets_base / 'human_pic'), name='human_pic_files')
app.mount('/files/DFM', StaticFiles(directory=_assets_base / 'DFM'), name='dfm_files')
app.mount('/files/m_blank', StaticFiles(directory=_assets_base / 'm_blank'), name='scene_files')

@app.get('/routes')
async def list_routes():
    return [r.path for r in app.router.routes]
