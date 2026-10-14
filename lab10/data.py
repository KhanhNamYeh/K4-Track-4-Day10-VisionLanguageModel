"""Nhãn YOLO → JSON grounding (thang 0–1000 hoặc pixel) và prompt (GUIDE bước 2)."""
import json

from scripts.labcommon import CLASSES

PROMPTS = {
    "norm1000": (
        "Detect road damage in this dashcam image. Classes: pothole, crack, manhole. "
        "Return ONLY a JSON list, no other text: "
        '[{"label": "<class>", "bbox_2d": [x1, y1, x2, y2]}]. '
        "Coordinates are integers on a 0-1000 scale relative to the image width and height "
        "(0,0 = top-left, 1000,1000 = bottom-right). Return [] if there is no damage."
    ),
    "pixel": (
        "Detect road damage in this dashcam image. Classes: pothole, crack, manhole. "
        "Return ONLY a JSON list, no other text: "
        '[{"label": "<class>", "bbox_2d": [x1, y1, x2, y2]}]. '
        "Coordinates are integer pixel positions in the {w}x{h} image "
        "(0,0 = top-left). Return [] if there is no damage."
    ),
}


def prompt_for(coord="norm1000", width=640, height=360):
    return PROMPTS[coord].replace("{w}", str(width)).replace("{h}", str(height))


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    xc, yc, w, h = yolo
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    box = [(xc - w / 2) * sx, (yc - h / 2) * sy, (xc + w / 2) * sx, (yc + h / 2) * sy]
    lim = (sx, sy, sx, sy)
    box = [min(max(int(round(v)), 0), m) for v, m in zip(box, lim)]
    return {"label": label, "bbox_2d": box}


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


assert set(CLASSES) == {"pothole", "crack", "manhole"}
