import io
import os
import uuid
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image
from torchvision import models
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt


APP_NAME = "TumorSight 360"
BASE_DIR = Path(__file__).resolve().parents[1]
YOLO_PATH = BASE_DIR / "best.pt"
UNET_PATH = BASE_DIR / "unet_busi.pth"
CLS_PATH = BASE_DIR / "resnet18_busi_cls.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE_SEG = 256
IMG_SIZE_CLS = 224
CLASS_NAMES = ["benign", "malignant"]
TMP_DIR = BASE_DIR / "webapp" / "tmp_uploads"
TMP_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB upload


def _case_state_path(case_id: str) -> Path:
    return TMP_DIR / f"{case_id}.json"


def load_case_state(case_id: str) -> dict:
    path = _case_state_path(case_id)
    if not path.exists():
        return {"case_id": case_id}
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # If corrupted, start fresh rather than hard-failing the UX.
        return {"case_id": case_id}


def save_case_state(case_id: str, patch: dict) -> dict:
    import json

    state = load_case_state(case_id)
    state.update(patch or {})
    _case_state_path(case_id).write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


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


def roc_curve_manual(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    P = int(np.sum(y_true == 1))
    N = int(np.sum(y_true == 0))
    if P == 0 or N == 0:
        raise ValueError("ROC needs both positive and negative samples.")

    tps = 0
    fps = 0
    fpr = [0.0]
    tpr = [0.0]
    for yt in y_true:
        if yt == 1:
            tps += 1
        else:
            fps += 1
        fpr.append(fps / N)
        tpr.append(tps / P)
    return np.array(fpr), np.array(tpr)


def auc_trapz(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    trap = getattr(np, "trapz", None)
    if callable(trap):
        return float(trap(y, x))
    # numpy>=2.0
    return float(np.trapezoid(y, x))


def build_roc_plot(fpr, tpr, auc_val):
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc_val:.3f})", color="#5DA5DA", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ResNet18 ROC (BUSI_Jpeg sample)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    return fig_to_data_uri(fig)


def compute_roc():
    """Compute the ROC curve fresh on every call (no caching)."""
    result = {"uri": None, "auc": None, "n": 0, "error": None}
    try:
        max_samples = int(os.getenv("ROC_MAX_SAMPLES", "200"))
        base_dir = BASE_DIR / "BUSI_Jpeg"
        pos_dir = base_dir / "malignant"
        neg_dir = base_dir / "benign"

        pos_paths = sorted(pos_dir.glob("*.png"))
        neg_paths = sorted(neg_dir.glob("*.png"))
        if not pos_paths or not neg_paths:
            raise FileNotFoundError("BUSI_Jpeg/benign and BUSI_Jpeg/malignant PNGs are required for ROC.")

        # Balance classes and cap.
        m = min(len(pos_paths), len(neg_paths), max_samples // 2 if max_samples > 1 else 1)
        pos_paths = pos_paths[:m]
        neg_paths = neg_paths[:m]

        y_true = []
        y_score = []

        # Prevent training-time artifacts like masks and non-image files by strict dirs + extension.
        for p in neg_paths:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            _, probs = predict_classification(rgb)
            y_true.append(0)
            y_score.append(float(probs[1]))  # P(malignant)

        for p in pos_paths:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            _, probs = predict_classification(rgb)
            y_true.append(1)
            y_score.append(float(probs[1]))  # P(malignant)

        if len(set(y_true)) < 2:
            raise ValueError("Not enough valid images to compute ROC.")

        fpr, tpr = roc_curve_manual(y_true, y_score)
        auc_val = auc_trapz(fpr, tpr)
        result.update({"uri": build_roc_plot(fpr, tpr, auc_val), "auc": float(auc_val), "n": int(len(y_true))})
    except Exception as e:
        result.update({"error": str(e)})

    return result


def save_uploaded_image(image_pil):
    case_id = str(uuid.uuid4())
    path = TMP_DIR / f"{case_id}.png"
    image_pil.save(path)
    return case_id


def load_case_image(case_id):
    path = TMP_DIR / f"{case_id}.png"
    if not path.exists():
        return None
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def cleanup_case(case_id):
    path = TMP_DIR / f"{case_id}.png"
    if path.exists():
        path.unlink()
    state_path = _case_state_path(case_id)
    if state_path.exists():
        state_path.unlink()


@app.route("/", methods=["GET"])
def index():
    message = request.args.get("message")
    return render_template("index.html", app_name=APP_NAME, message=message)


@app.route("/analyze", methods=["POST"])
def analyze():
    # This endpoint handles the full flow (YOLO -> Segmentation -> Classification)
    # using a single URL. The UI advances by posting `stage`, `case_id`, and `decision`.

    stage = request.form.get("stage")
    case_id = request.form.get("case_id")
    decision = request.form.get("decision")
    file = request.files.get("image")

    # ----- Stage 0: new upload (no stage/case_id yet) -----
    if file is not None and file.filename:
        consent = request.form.get("consent")
        if consent != "yes":
            return render_template(
                "index.html",
                app_name=APP_NAME,
                error="You must read and accept the User Notice & Consent before proceeding.",
            )

        image_pil = Image.open(file.stream).convert("RGB")
        img_rgb = np.array(image_pil)
        case_id = save_uploaded_image(image_pil)

        # YOLO
        yolo_box, yolo_conf, yolo_num_boxes, yolo_used_th, yolo_variant = predict_yolo(img_rgb)
        img_area = img_rgb.shape[0] * img_rgb.shape[1]
        yolo_area_pct = None
        if yolo_box is not None:
            yolo_area = max(0.0, yolo_box[2] - yolo_box[0]) * max(0.0, yolo_box[3] - yolo_box[1])
            yolo_area_pct = float((yolo_area / img_area) * 100.0)

        detection_overlay_uri = build_detection_plot(img_rgb, yolo_box)
        input_preview_uri = "data:image/png;base64," + base64_encode(
            cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))[1].tobytes()
        )

        save_case_state(
            case_id,
            {
                "yolo": {
                    "box": yolo_box.tolist() if yolo_box is not None else None,
                    "conf": float(yolo_conf),
                    "num_boxes": int(yolo_num_boxes),
                    "used_th": float(yolo_used_th) if yolo_used_th is not None else None,
                    "variant": str(yolo_variant),
                    "area_pct": float(yolo_area_pct) if yolo_area_pct is not None else None,
                    "decision": None,
                },
                "segmentation": {"decision": None},
            },
        )

        return render_template(
            "analyze.html",
            app_name=APP_NAME,
            stage="yolo",
            case_id=case_id,
            input_preview_uri=input_preview_uri,
            detection_overlay_uri=detection_overlay_uri,
            yolo_conf=yolo_conf,
            yolo_num_boxes=yolo_num_boxes,
            yolo_used_th=yolo_used_th,
            yolo_variant=yolo_variant,
            yolo_area_pct=yolo_area_pct,
        )

    # ----- Stage transitions (same URL) -----
    if not case_id or not stage:
        return redirect(url_for("index", message="Session expired. Please upload again."))

    img_rgb = load_case_image(case_id)
    if img_rgb is None:
        return redirect(url_for("index", message="Image not found. Please upload again."))

    state = load_case_state(case_id)
    y = state.get("yolo", {}) if isinstance(state, dict) else {}
    yolo_box = np.array(y["box"], dtype=float) if y.get("box") is not None else None
    detection_overlay_uri = build_detection_plot(img_rgb, yolo_box)
    input_preview_uri = "data:image/png;base64," + base64_encode(
        cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))[1].tobytes()
    )

    if stage == "yolo":
        if decision == "decline":
            save_case_state(case_id, {"yolo": {**y, "decision": "declined"}})
            cleanup_case(case_id)
            return redirect(url_for("index", message="YOLO result declined. Please upload a new image."))

        save_case_state(case_id, {"yolo": {**y, "decision": "accepted"}})

        seg_prob, seg_mask = predict_segmentation(img_rgb)
        seg_overlay_uri = build_segmentation_overlay_plot(img_rgb, seg_mask)
        seg_prob_uri = build_probability_plot(seg_prob)
        img_area = img_rgb.shape[0] * img_rgb.shape[1]
        seg_area = int(np.sum(seg_mask > 0))
        seg_area_pct = (seg_area / img_area) * 100.0
        save_case_state(
            case_id,
            {
                "segmentation": {
                    "area": int(seg_area),
                    "area_pct": float(seg_area_pct),
                    "decision": None,
                }
            },
        )

        return render_template(
            "analyze.html",
            app_name=APP_NAME,
            stage="segmentation",
            case_id=case_id,
            input_preview_uri=input_preview_uri,
            detection_overlay_uri=detection_overlay_uri,
            yolo_conf=y.get("conf", 0.0),
            yolo_num_boxes=y.get("num_boxes", 0),
            yolo_used_th=y.get("used_th"),
            yolo_variant=y.get("variant", "unknown"),
            yolo_area_pct=y.get("area_pct"),
            seg_overlay_uri=seg_overlay_uri,
            seg_prob_uri=seg_prob_uri,
            seg_area=seg_area,
            seg_area_pct=seg_area_pct,
        )

    if stage == "segmentation":
        seg_state = state.get("segmentation", {}) if isinstance(state, dict) else {}
        if decision == "decline":
            save_case_state(case_id, {"segmentation": {**seg_state, "decision": "declined"}})
            cleanup_case(case_id)
            return redirect(url_for("index", message="Segmentation result declined. Please upload a new image."))

        save_case_state(case_id, {"segmentation": {**seg_state, "decision": "accepted"}})

        seg_prob, seg_mask = predict_segmentation(img_rgb)
        seg_overlay_uri = build_segmentation_overlay_plot(img_rgb, seg_mask)
        seg_prob_uri = build_probability_plot(seg_prob)
        img_area = img_rgb.shape[0] * img_rgb.shape[1]
        seg_area = int(np.sum(seg_mask > 0))
        seg_area_pct = (seg_area / img_area) * 100.0

        cls_idx, cls_probs = predict_classification(img_rgb)
        cls_bar_uri = build_class_plot(cls_probs)
        roc = compute_roc()

        cleanup_case(case_id)

        return render_template(
            "analyze.html",
            app_name=APP_NAME,
            stage="classification",
            case_id=case_id,
            input_preview_uri=input_preview_uri,
            detection_overlay_uri=detection_overlay_uri,
            yolo_conf=y.get("conf", 0.0),
            yolo_num_boxes=y.get("num_boxes", 0),
            yolo_used_th=y.get("used_th"),
            yolo_variant=y.get("variant", "unknown"),
            yolo_area_pct=y.get("area_pct"),
            seg_overlay_uri=seg_overlay_uri,
            seg_prob_uri=seg_prob_uri,
            seg_area=seg_area,
            seg_area_pct=seg_area_pct,
            cls_bar_uri=cls_bar_uri,
            cls_pred=CLASS_NAMES[cls_idx],
            cls_prob_benign=float(cls_probs[0]),
            cls_prob_malignant=float(cls_probs[1]),
            roc_uri=roc.get("uri"),
            roc_auc=roc.get("auc"),
            roc_n=roc.get("n"),
            roc_error=roc.get("error"),
        )

    return redirect(url_for("index", message="Unknown stage. Please upload again."))




if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5001")))
    debug = os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)
