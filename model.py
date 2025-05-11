import os, torch, numpy as np
from PIL import Image, ImageOps
from torchvision import transforms, models
from collections import OrderedDict
import torch.nn as nn
import cv2
import mediapipe as mp
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 경로 (네 환경에 맞게 수정)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

regression_ckpt = os.path.join(BASE_DIR, "checkpoint")
skin_ckpt = os.path.join(regression_ckpt, "skin_type", "state_dict.bin")


# 트랜스폼
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

skin_label_names = ['건성', '복합건성', '복합지성', '중성', '지성']
regression_num_output = [1, 2, 0, 0, 0, 3, 3, 0, 2]
area_label = {0: "전체", 1: "이마", 2: "미간", 3: "왼쪽 눈가", 4: "오른쪽 눈가", 5: "왼쪽 볼", 6: "오른쪽 볼", 7: "입술", 8: "턱"}
reg_desc = {0: ["색소침착 개수"], 1: ["수분", "탄력"], 5: ["수분", "탄력", "모공 개수"], 6: ["수분", "탄력", "모공 개수"], 8: ["수분", "탄력"]}
restore_stats = {
    1: {"수분": (60.6, 10.1), "탄력": (48.7, 11.9)},
    5: {"수분": (60.6, 10.1), "탄력": (48.7, 11.9), "모공 개수": "log"},
    6: {"수분": (60.2, 9.6),  "탄력": (49.3, 12.1), "모공 개수": "log"},
    8: {"수분": (61.3, 10.0), "탄력": (47.5, 12.0)},
    0: {"색소침착 개수": 300}
}

mp_face_mesh = mp.solutions.face_mesh
REGION_LANDMARKS = {
    0: list(range(468)), 1: [10, 67, 69, 71, 109, 151, 337, 338, 297],
    2: [168, 6, 197, 195, 5, 4], 3: [130, 133, 160, 159, 158],
    4: [359, 362, 386, 385, 384], 5: [205, 50, 187, 201, 213],
    6: [425, 280, 411, 427, 434], 7: [13, 14, 17, 84, 181],
    8: [152, 377, 400, 378, 379]
}

def crop_regions_by_ratio(pil_img):
    img = np.array(pil_img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    regions = [None] * 9
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            raise ValueError("❗ 얼굴을 찾을 수 없습니다.")
        landmarks = results.multi_face_landmarks[0].landmark
        points = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
        face_x1, face_y1 = np.min(points, axis=0)
        face_x2, face_y2 = np.max(points, axis=0)
        face_w, face_h = face_x2 - face_x1, face_y2 - face_y1
        for idx, lm_indices in REGION_LANDMARKS.items():
            pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in lm_indices])
            cx, cy = np.mean(pts, axis=0)
            if idx == 8: cx -= face_w * 0.15
            if idx == 1: box_w, box_h = int(face_w * 0.70), int(face_h * 0.3); cy -= box_h * 0.2
            elif idx == 2: box_w, box_h = int(face_w * 0.35), int(face_h * 0.15); cy -= box_h * 2.5
            else: box_w, box_h = int(face_w * 0.28), int(face_h * 0.25)
            x1, y1 = max(int(cx - box_w / 2), 0), max(int(cy - box_h / 2), 0)
            x2, y2 = min(int(cx + box_w / 2), w), min(int(cy + box_h / 2), h)
            crop = img[y1:y2, x1:x2]
            regions[idx] = Image.fromarray(crop)
    return regions

# 모델 로딩
reg_models = []
for idx, out_dim in enumerate(regression_num_output):
    if out_dim == 0:
        reg_models.append(None)
        continue
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, out_dim)
    ckpt_path = os.path.join(regression_ckpt, str(idx), "state_dict.bin")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        if "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=False)
        model.eval()
        reg_models.append(model.to(device))
    else:
        reg_models.append(None)

skin_model = models.resnet50(weights=None)
skin_model.fc = nn.Linear(skin_model.fc.in_features, len(skin_label_names))
skin_model.load_state_dict(torch.load(skin_ckpt, map_location=device))
skin_model.eval()
skin_model = skin_model.to(device)

# 분석 함수
def model_image(image: Image.Image) -> dict:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    regions = crop_regions_by_ratio(image)
    result = {}

    for idx in range(9):
        if reg_models[idx] is None or regions[idx] is None or idx in [3, 4]:
            continue
        crop_tensor = transform(regions[idx]).unsqueeze(0).to(device)
        with torch.no_grad():
            output = reg_models[idx](crop_tensor).squeeze().cpu().numpy()
        if output.ndim == 0:
            output = [output]
        sub_result = {}
        for i, val in enumerate(output):
            label = reg_desc[idx][i]
            if label == "모공 개수" and restore_stats[idx].get(label) == "log":
                val = np.clip(np.exp(val) - 1, 0, 2500)
            elif isinstance(restore_stats[idx].get(label), tuple):
                mean, std = restore_stats[idx][label]
                val = val * std + mean
            elif label == "색소침착 개수":
                val *= 300
            sub_result[label] = round(float(val), 2)
        result[area_label[idx]] = sub_result

    # 피부 타입 분류
    if regions[0] is not None:
        overall_tensor = transform(regions[0]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.argmax(skin_model(overall_tensor), dim=1).item()
            result["피부 타입"] = skin_label_names[pred]

    ordered_result = OrderedDict()
    for key in ["전체", "이마", "왼쪽 볼", "오른쪽 볼", "턱", "피부 타입"]:
        if key in result:
            ordered_result[key] = result[key]

    return ordered_result

