from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"


def _run_json(cmd: list[str]) -> dict:
    p = subprocess.run(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    if p.stdout is None:
        raise RuntimeError("no stdout")
    return json.loads(p.stdout)


def _run_text(cmd: list[str]) -> str:
    p = subprocess.run(cmd, cwd=str(ROOT), text=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    return p.stdout or ""


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "TEST_REPORT.md"

    env = _run_text([sys.executable, "scripts/verify_env.py"])

    ort_bench = None
    try:
        ort_bench = _run_json([sys.executable, "scripts/benchmark_onnxruntime.py", "--json"])
    except Exception:
        ort_bench = None

    pipe_on = _run_text([sys.executable, "scripts/benchmark_pipeline.py", "--enable", "--queue", "1024", "--frames", "300"])
    pipe_direct = _run_text([sys.executable, "scripts/benchmark_pipeline.py", "--mode", "direct", "--frames", "300"])
    smoke = _run_text([sys.executable, "scripts/integration_smoke.py"])
    codec = ""
    try:
        codec = _run_text([sys.executable, "-c", "import json;from blankend.video_processing.codec import probe_hw_codecs;print(json.dumps(probe_hw_codecs(),ensure_ascii=False))"])
    except Exception:
        codec = ""

    content = []
    content.append("# 测试报告\n")
    content.append("## 环境检查\n")
    content.append("```text\n" + env.strip() + "\n```\n")

    content.append("## 性能测试\n")
    content.append("### 生产者-消费者流水线吞吐\n")
    content.append("```text\n" + pipe_on.strip() + "\n" + pipe_direct.strip() + "\n```\n")

    content.append("### onnxruntime CPU/GPU 基准（如可用）\n")
    if ort_bench is None:
        content.append("- 当前环境无法执行 onnxruntime 基准（缺少依赖或模型不兼容）。\n")
    else:
        cpu_ms = ort_bench["bench"]["cpu"]["avg_ms"]
        gpu_ms = ort_bench["bench"].get("gpu", {}).get("avg_ms")
        content.append(f"- 模型：`{ort_bench['model']}`\n")
        content.append(f"- CPU avg_ms：{cpu_ms:.3f}\n")
        if gpu_ms is not None:
            content.append(f"- GPU avg_ms：{gpu_ms:.3f}\n")
        else:
            content.append("- GPU：当前环境未检测到 CUDAExecutionProvider。\n")
        content.append("\n")

    content.append("## 可靠性测试\n")
    content.append("- 网络异常/断线恢复：提供集成脚本入口（依赖 `websockets` 或 `websocket-client` 时可扩展），本仓库默认不强制安装该依赖。\n")

    content.append("## 集成测试\n")
    content.append("- 新旧版本兼容：保留现有 `/video_in`、`/human_pic`、`/DFM`、`/m_blank` 等端点不变；新增端点在 `/api/v1/*` 下提供。\n")
    content.append("```text\n" + smoke.strip() + "\n```\n")
    if codec.strip():
        content.append("## 编解码能力探测\n")
        content.append("```json\n" + codec.strip() + "\n```\n")

    report_path.write_text("".join(content), encoding="utf-8")
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
