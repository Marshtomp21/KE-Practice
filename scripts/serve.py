"""启动 Web 服务。用法：python scripts/serve.py [--port 8000] [--reload]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

from src.core.config import load_settings


def main() -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=settings.get("api.host", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(settings.get("api.port", 8000)))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    print(f"打开 http://{args.host}:{args.port}/ 使用三栏界面")
    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=str(settings.get("runtime.log_level", "info")).lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
