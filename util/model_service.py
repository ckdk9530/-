import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple
from PIL import Image

# Ensure CUDA works with multiprocessing
def ensure_spawn_start_method() -> None:
    """Configure multiprocessing to use the 'spawn' start method."""
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)


class _BaseProcess(mp.Process):
    def __init__(self):
        super().__init__()
        self.in_q: mp.Queue = mp.Queue()
        self.out_q: mp.Queue = mp.Queue()

    def predict(self, data):
        self.in_q.put(data)
        return self.out_q.get()

    def shutdown(self):
        self.in_q.put(None)
        self.join()


class YoloWorker(_BaseProcess):
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__()
        self.model_path = model_path
        self.device = device

    def run(self):
        from OmniParser.util.utils import get_yolo_model
        model = get_yolo_model(self.model_path, device=self.device)
        try:
            while True:
                data = self.in_q.get()
                if data is None:
                    break
                paths: List[str] = data
                imgs = [Image.open(p).convert("RGB") for p in paths]
                preds = model.predict(imgs, batch=len(imgs), verbose=False)
                outputs = []
                for r in preds:
                    boxes = r.boxes.xyxy
                    conf = r.boxes.conf
                    phrases = [str(i) for i in range(len(boxes))]
                    outputs.append((boxes, conf, phrases))
                self.out_q.put(outputs)
        except KeyboardInterrupt:
            pass


class OCRWorker(_BaseProcess):
    def __init__(self, use_gpu: bool = True):
        super().__init__()
        self.use_gpu = use_gpu

    def run(self):
        from OmniParser.util.utils import _get_paddle_ocr, check_ocr_box
        ocr = _get_paddle_ocr(use_gpu=self.use_gpu)
        try:
            while True:
                data = self.in_q.get()
                if data is None:
                    break
                idx, path = data
                img = Image.open(path).convert("RGB")
                result = check_ocr_box(
                    img,
                    output_bb_format="xyxy",
                    use_paddleocr=True,
                    paddle_ocr=ocr,
                    easyocr_args={"paragraph": False, "text_threshold": 0.9},
                )
                self.out_q.put((idx, result))
        except KeyboardInterrupt:
            pass



