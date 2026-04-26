import io
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request
from PIL import Image
from torchvision import models
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt


APP_NAME = "TumorSight 360"
BASE_DIR = Path(__file__).resolve().parents[1]
YOLO_PATH = BASE_DIR / "yolo26n.pt"
UNET_PATH = BASE_DIR / "unet_busi.pth"
CLS_PATH = BASE_DIR / "resnet18_busi_cls.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE_SEG = 256
IMG_SIZE_CLS = 224
CLASS_NAMES = ["benign", "malignant"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB upload


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        c1 = self.down1(x)
        c2 = self.down2(self.pool1(c1))
        c3 = self.down3(self.pool2(c2))
        bn = self.bottleneck(self.pool3(c3))
        u3 = self.conv3(torch.cat([self.up3(bn), c3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), c1], dim=1))
        return self.out(u1)


def load_models():
    if not YOLO_PATH.exists() or not UNET_PATH.exists() or not CLS_PATH.exists():
        missing = [str(p.name) for p in [YOLO_PATH, UNET_PATH, CLS_PATH] if not p.exists()]
        raise FileNotFoundError(f"Missing model files: {', '.join(missing)}")

    yolo_model = YOLO(str(YOLO_PATH))

    unet_model = UNet().to(DEVICE)
    unet_model.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
    unet_model.eval()

    cls_model = models.resnet18(weights=None)
    cls_model.fc = nn.Linear(cls_model.fc.in_features, 2)
    cls_model.load_state_dict(torch.load(CLS_PATH, map_location=DEVICE))
    cls_model = cls_model.to(DEVICE)
    cls_model.eval()
    return yolo_model, unet_model, cls_model


yolo_model, unet_model, cls_model = load_models()


def fig_to_data_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    b64 = base64_encode(buf.getvalue())
    return f"data:image/png;base64,{b64}"


def base64_encode(data):
    import base64

    return base64.b64encode(data).decode("utf-8")


def compute_iou_box(a, b):
    x_a = max(a[0], b[0])
    y_a = max(a[1], b[1])
    x_b = min(a[2], b[2])
    y_b = min(a[3], b[3])
    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mask_to_bbox(mask_bin):
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return np.array([x, y, x + w, y + h], dtype=float)


def make_enhanced_rgb(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Light sharpening
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)


def predict_yolo(img_rgb):
    thresholds = [0.25, 0.10, 0.03, 0.01, 0.003]
    variants = [
        ("original", img_rgb),
        ("enhanced", make_enhanced_rgb(img_rgb)),
    ]

    best_box = None
    best_conf = -1.0
    best_count = 0
    best_th = None
    best_variant = "none"

    for variant_name, variant_img in variants:
        for th in thresholds:
            results = yolo_model.predict(source=variant_img, conf=th, imgsz=960, verbose=False)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue

            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            idx = int(np.argmax(confs))
            conf_val = float(confs[idx])
            if conf_val > best_conf:
                best_box = xyxy[idx]
                best_conf = conf_val
                best_count = int(len(boxes))
                best_th = th
                best_variant = variant_name

    if best_box is None:
        return None, 0.0, 0, None, "none"
    return best_box, best_conf, best_count, best_th, best_variant


def predict_segmentation(img_rgb):
    resized = cv2.resize(img_rgb, (IMG_SIZE_SEG, IMG_SIZE_SEG), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(unet_model(x))[0, 0].cpu().numpy()
    prob_full = cv2.resize(prob, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask_bin = (prob_full >= 0.5).astype(np.uint8) * 255
    return prob_full, mask_bin


def predict_classification(img_rgb):
    img = cv2.resize(img_rgb, (IMG_SIZE_CLS, IMG_SIZE_CLS), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = cls_model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def build_detection_plot(img_rgb, yolo_box):
    vis = img_rgb.copy()

    if yolo_box is not None:
        x1, y1, x2, y2 = yolo_box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(vis)
    ax.set_title("YOLO Detection")
    ax.axis("off")
    return fig_to_data_uri(fig)


def build_segmentation_overlay_plot(img_rgb, seg_mask):
    vis = img_rgb.copy()
    seg_col = np.zeros_like(vis)
    seg_col[:, :, 0] = (seg_mask > 0).astype(np.uint8) * 255
    vis = cv2.addWeighted(vis, 1.0, seg_col, 0.30, 0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(vis)
    ax.set_title("U-Net Segmentation Overlay")
    ax.axis("off")
    return fig_to_data_uri(fig)


def build_probability_plot(seg_prob):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(seg_prob, cmap="magma")
    ax.set_title("U-Net Probability Heatmap")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig_to_data_uri(fig)


def build_class_plot(probs):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(CLASS_NAMES, probs, color=["#5DA5DA", "#F17CB0"])
    ax.set_ylim(0, 1)
    ax.set_title("Classification Probabilities (ResNet18)")
    ax.set_ylabel("Probability")
    ax.grid(axis="y", alpha=0.3)
    return fig_to_data_uri(fig)


def metrics_with_optional_gt(pred_mask, gt_mask_optional):
    if gt_mask_optional is None:
        return {
            "dice": None,
            "iou": None,
            "precision": None,
            "recall": None,
            "note": "Ground-truth mask not provided. True segmentation metrics unavailable.",
        }

    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask_optional > 0).astype(np.uint8)

    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "note": "Computed against uploaded ground-truth mask.",
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", app_name=APP_NAME)


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    consent = request.form.get("consent")

    if consent != "yes":
        return render_template(
            "index.html",
            app_name=APP_NAME,
            error="You must read and accept the User Notice & Consent before proceeding.",
        )

    if file is None or file.filename == "":
        return render_template("index.html", app_name=APP_NAME, error="Please upload an image.")

    image_pil = Image.open(file.stream).convert("RGB")
    img_rgb = np.array(image_pil)

    gt_mask = None
    gt_file = request.files.get("mask")
    if gt_file is not None and gt_file.filename:
        gt_pil = Image.open(gt_file.stream).convert("L")
        gt_mask = np.array(gt_pil)
        if gt_mask.shape[:2] != img_rgb.shape[:2]:
            gt_mask = cv2.resize(gt_mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    # YOLO
    yolo_box, yolo_conf, yolo_num_boxes, yolo_used_th, yolo_variant = predict_yolo(img_rgb)

    # U-Net segmentation
    seg_prob, seg_mask = predict_segmentation(img_rgb)
    seg_bbox = mask_to_bbox(seg_mask)

    # ResNet classification
    cls_idx, cls_probs = predict_classification(img_rgb)

    # Derived cross-model metrics
    img_area = img_rgb.shape[0] * img_rgb.shape[1]
    seg_area = int(np.sum(seg_mask > 0))
    seg_area_pct = (seg_area / img_area) * 100.0
    yolo_area_pct = None
    box_mask_iou = None
    if yolo_box is not None:
        yolo_area = max(0.0, yolo_box[2] - yolo_box[0]) * max(0.0, yolo_box[3] - yolo_box[1])
        yolo_area_pct = float((yolo_area / img_area) * 100.0)
        if seg_bbox is not None:
            box_mask_iou = float(compute_iou_box(yolo_box, seg_bbox))

    seg_metrics = metrics_with_optional_gt(seg_mask, gt_mask)

    detection_overlay_uri = build_detection_plot(img_rgb, yolo_box)
    seg_overlay_uri = build_segmentation_overlay_plot(img_rgb, seg_mask)
    seg_prob_uri = build_probability_plot(seg_prob)
    cls_bar_uri = build_class_plot(cls_probs)

    input_preview_uri = "data:image/png;base64," + base64_encode(cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))[1].tobytes())

    return render_template(
        "result.html",
        app_name=APP_NAME,
        input_preview_uri=input_preview_uri,
        detection_overlay_uri=detection_overlay_uri,
        seg_overlay_uri=seg_overlay_uri,
        seg_prob_uri=seg_prob_uri,
        cls_bar_uri=cls_bar_uri,
        yolo_conf=yolo_conf,
        yolo_num_boxes=yolo_num_boxes,
        yolo_used_th=yolo_used_th,
        yolo_variant=yolo_variant,
        yolo_area_pct=yolo_area_pct,
        box_mask_iou=box_mask_iou,
        seg_area=seg_area,
        seg_area_pct=seg_area_pct,
        seg_metrics=seg_metrics,
        cls_pred=CLASS_NAMES[cls_idx],
        cls_prob_benign=float(cls_probs[0]),
        cls_prob_malignant=float(cls_probs[1]),
    )


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5001")))
    debug = os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)
