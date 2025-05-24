"""GPU-OCR sub-process
-------------------
用法：
    python -m OmniParser.util.ocr_worker /path/to/img.jpg
輸出：
    {"ocr_text": [...], "ocr_bbox": [[x0,y0,x1,y1], ...]}
"""
import os, sys, json, gc, tempfile
from PIL import Image
from OmniParser.util.utils import check_ocr_box, _get_paddle_ocr

# ─ GPU 設定（想用 CPU 可以改 use_gpu=False） ───────────
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
DEVICE = "cuda"
PADDLE_OCR = _get_paddle_ocr(use_gpu=(DEVICE=="cuda"), rec_batch_num=2)

def run(img_path):
    img = Image.open(img_path).convert("RGB")
    (text, bbox), _ = check_ocr_box(
        img,
        display_img=False,
        output_bb_format="xyxy",
        paddle_ocr=PADDLE_OCR,
        use_paddleocr=True,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
    )
    print(json.dumps({"ocr_text": text, "ocr_bbox": bbox}, ensure_ascii=False))
    sys.stdout.flush()
    # 結束前盡可能清空 Pool
    import paddle
    gc.collect(); paddle.device.cuda.empty_cache()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Need image path")
    run(sys.argv[1])

