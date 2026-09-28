# Chest disease detection model (to be added)

Put here:
- your model: `best.onnx` (YOLOv8/11 exported with `yolo export model=best.pt format=onnx`) **or** `best.pt` (needs `ultralytics`)
- `config.json` — copy `config.example.json` and edit it

| field | meaning |
|---|---|
| `framework` | `yolo_onnx` (default), `ultralytics`, or `classifier_onnx` (no boxes) |
| `model_file` | file name inside this folder |
| `labels` | class names, same order as training (optional for `ultralytics`) |
| `conf_threshold` / `iou_threshold` | default detection confidence / NMS IoU |
| `input_size` | `[h, w]`, optional — read from the ONNX model when omitted |

Classifier-only extras: `color_mode` (RGB/L), `normalize` (zero_one/minus_one_one/imagenet/none),
`channels_first`, `normal_labels` (classes meaning "no abnormality", default: Normal / No Finding).

Files above 100 MB need Git LFS (`git lfs track "*.onnx" "*.pt"`).
