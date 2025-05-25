"""GPU memory utilities for PaddleOCR"""

from __future__ import annotations

try:  # 避免在缺少套件時失敗
    import torch
except Exception:  # pragma: no cover - 避免測試環境缺套件
    torch = None

try:  # pragma: no cover - 同上
    import paddle
except Exception:  # pragma: no cover - 同上
    paddle = None


def debug_gpu_memory(func_name: str) -> None:
    """Print current GPU memory usage for debug purpose."""
    torch_mem = "N/A"
    paddle_mem = "N/A"
    try:
        if torch and torch.cuda.is_available():
            torch_mem = int(torch.cuda.memory_allocated() / 1024 / 1024)
    except Exception:  # pragma: no cover - 避免任何意外錯誤
        pass
    try:
        if paddle and paddle.is_compiled_with_cuda():
            free, total = paddle.device.cuda.mem_get_info()
            paddle_mem = int((total - free) / 1024 / 1024)
    except Exception:  # pragma: no cover - 避免任何意外錯誤
        pass
    print(f"[DEBUG] {func_name}: torch_gpu={torch_mem}MB paddle_gpu={paddle_mem}MB")


def release_ocr_gpu_cache(ocr_reader):
    """針對 PaddleOCR 三個 predictor 逐一釋放 GPU 暫存"""
    debug_gpu_memory("release_ocr_gpu_cache (before)")
    for name in ("text_detector", "text_classifier", "text_recognizer"):
        comp = getattr(ocr_reader, name, None)
        if comp is None or not hasattr(comp, "predictor"):
            continue
        pred = comp.predictor
        try:
            pred.clear_intermediate_tensor()
            debug_gpu_memory(f"release_ocr_gpu_cache clear {name}")
        except Exception:
            pass
        try:
            pred.try_shrink_memory()
            debug_gpu_memory(f"release_ocr_gpu_cache shrink {name}")
        except Exception:
            pass

