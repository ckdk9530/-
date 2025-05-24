"""GPU memory utilities for PaddleOCR"""


def release_ocr_gpu_cache(ocr_reader):
    """針對 PaddleOCR 三個 predictor 逐一釋放 GPU 暫存"""
    for name in ("text_detector", "text_classifier", "text_recognizer"):
        comp = getattr(ocr_reader, name, None)
        if comp is None or not hasattr(comp, "predictor"):
            continue
        pred = comp.predictor
        try:
            pred.clear_intermediate_tensor()
        except Exception:
            pass
        try:
            pred.try_shrink_memory()
        except Exception:
            pass

