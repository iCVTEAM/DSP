import os
import json
from PIL import Image, ImageDraw, ImageFont

def draw_box_desc_on_blank(obboxes, prompt, image_size=(512, 512)):
    color_list = ['red', 'blue', 'yellow', 'purple', 'green', 
                  'black', 'brown', 'orange', 'white', 'gray']

    width, height = image_size
    pil_img = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype('../../fonts/GILI____.TTF', 56)
    except:
        font = ImageFont.load_default()

    for obj_box, text_desc in zip(obboxes, prompt):
        if not any(obj_box):
            continue

        fill_color = next((color for color in text_desc.split(' ') if color in color_list), 'black')
        text = text_desc.split(',')[0]

        polygon_points = [(x * width, y * height) for x, y in zip(obj_box[::2], obj_box[1::2])]
        draw.polygon(polygon_points, outline=fill_color, width=3)

        x_min, y_min = polygon_points[0]
        draw.text((int(x_min), int(y_min)), text, fill=fill_color, font=font)

    return pil_img


def process_jsonl(jsonl_paths, output_dir, image_size=(512, 512)):
    os.makedirs(output_dir, exist_ok=True)

    for jsonl_path in jsonl_paths:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                obboxes = data.get("obboxes", [])
                captions = data.get("captions", [])

                img = draw_box_desc_on_blank(obboxes, captions[1:], image_size=image_size)

                file_name = os.path.basename(data["file_name"])
                save_path = os.path.join(output_dir, file_name)
                img.save(save_path)
                print(f"Saved {save_path}")


PROJECT_DIR = os.getenv('DSP_PROJECT_DIR', '/path/to/DSP_PROJECT_DIR') # Set this manually if the environment variable is unavailable

if __name__ == "__main__":
    jsonl_paths = [
        os.path.join(PROJECT_DIR, 'data/EXDARK/metadatas/data_setting1/val_novel_bus.jsonl'),
        os.path.join(PROJECT_DIR, 'data/EXDARK/metadatas/data_setting1/val_novel_dog.jsonl'),
        os.path.join(PROJECT_DIR, 'data/EXDARK/metadatas/data_setting1/val_novel_motorbike.jsonl'),
        os.path.join(PROJECT_DIR, 'data/EXDARK/metadatas/data_setting1/val_novel_table.jsonl'),
        os.path.join(PROJECT_DIR, 'data/EXDARK/metadatas/data_setting1/test_novel_bus.jsonl'),
        os.path.join(PROJECT_DIR, 'data/EXDARK/metadatas/data_setting1/test_novel_dog.jsonl'),
        os.path.join(PROJECT_DIR, 'data/EXDARK/metadatas/data_setting1/test_novel_motorbike.jsonl'),
        os.path.join(PROJECT_DIR, 'data/EXDARK/metadatas/data_setting1/test_novel_table.jsonl'),
    ]
    output_dir = os.path.join(PROJECT_DIR, "data/EXDARK/bboxes")
    process_jsonl(jsonl_paths, output_dir, image_size=(512, 512))