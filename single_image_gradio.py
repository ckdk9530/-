from typing import Optional

import gradio as gr
import torch
from PIL import Image
import io, base64

from OmniParser.util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
)

# 初始化模型
yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(
    model_name='florence2',
    model_name_or_path='weights/icon_caption_florence',
)

MARKDOWN = """# OmniParser 單張圖片測試頁"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process(
    image_input: Image.Image,
    box_threshold: float,
    iou_threshold: float,
    use_paddleocr: bool,
    imgsz: int,
) -> Optional[Image.Image]:
    box_overlay_ratio = image_input.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, _ = check_ocr_box(
        image_input,
        display_img=False,
        output_bb_format='xyxy',
        goal_filtering=None,
        easyocr_args={'paragraph': False, 'text_threshold': 0.9},
        use_paddleocr=use_paddleocr,
    )
    text, ocr_bbox = ocr_bbox_rslt
    dino_labled_img, _, parsed_content_list = get_som_labeled_img(
        image_input,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    )
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    parsed_content_list = '\n'.join(
        f'icon {i}: {v}' for i, v in enumerate(parsed_content_list)
    )
    return image, parsed_content_list


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image_input_component = gr.Image(type='pil', label='Upload image')
            box_threshold_component = gr.Slider(
                label='Box Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.05
            )
            iou_threshold_component = gr.Slider(
                label='IOU Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.1
            )
            use_paddleocr_component = gr.Checkbox(label='Use PaddleOCR', value=True)
            imgsz_component = gr.Slider(
                label='Icon Detect Image Size', minimum=640, maximum=1920, step=32, value=640
            )
            submit_button_component = gr.Button(value='Submit', variant='primary')
        with gr.Column():
            image_output_component = gr.Image(type='pil', label='Image Output')
            text_output_component = gr.Textbox(label='Parsed screen elements', placeholder='Text Output')

    submit_button_component.click(
        fn=process,
        inputs=[
            image_input_component,
            box_threshold_component,
            iou_threshold_component,
            use_paddleocr_component,
            imgsz_component,
        ],
        outputs=[image_output_component, text_output_component],
    )

demo.launch(share=True, server_port=7861, server_name='0.0.0.0')
