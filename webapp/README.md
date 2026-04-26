# TumorSight 360 (Flask App)

TumorSight 360 is a detailed medical imaging demo app that runs:

- YOLO detection (`best.pt`)
- U-Net segmentation (`unet_busi.pth`)
- ResNet18 classification (`resnet18_busi_cls.pth`)

on an uploaded tumor image.

## Features

- Upload image and get full combined report
- YOLO metrics:
  - best confidence
  - number of boxes
  - box area percentage
  - YOLO-box vs segmentation-box IoU
- Segmentation outputs:
  - mask overlay figure
  - probability heatmap
  - segmented area stats
  - optional true Dice/IoU/Precision/Recall (if ground-truth mask uploaded)
- Classification outputs:
  - predicted class (benign/malignant)
  - class probabilities
  - probability bar chart

## Run

From project root:

```bash
source .venv/bin/activate
pip install -r webapp/requirements.txt
python webapp/app.py
```

Then open:

- http://127.0.0.1:5001

## Docker (for Coolify)

From project root:

```bash
docker build -t tumorsight360 .
docker run --rm -p 5001:5001 -e PORT=5001 tumorsight360
```

Then open:

- http://127.0.0.1:5001

## Coolify Deployment

1. Push this project to GitHub.
2. In Coolify, create a **New Resource -> Application** from your repository.
3. Select **Dockerfile** as build pack (root `Dockerfile`).
4. Set port to `5001` (or provide env `PORT` and use that in Coolify).
5. Deploy.

Recommended env vars:

- `PORT=5001`
- `FLASK_DEBUG=false`

## Notes

- The app expects model files in project root:
  - `best.pt`
  - `unet_busi.pth`
  - `resnet18_busi_cls.pth`
- For true segmentation quality metrics, upload a matching ground-truth mask.
