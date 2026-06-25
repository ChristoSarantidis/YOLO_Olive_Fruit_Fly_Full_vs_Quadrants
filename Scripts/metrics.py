#!/usr/bin/env python3
# yolo_txt_metrics.py
# Compute Precision/Recall/F1/mAP50 from YOLO .txt labels folders (GT vs Predictions)

import os
import glob
import argparse
from collections import defaultdict
import numpy as np


def yolo_xywh_to_xyxy(box):
    """box: (xc, yc, w, h) normalized [0,1] -> (x1,y1,x2,y2) clamped [0,1]"""
    xc, yc, w, h = box
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    # clamp
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou_xyxy(a, b):
    """IoU between two boxes in xyxy"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def read_yolo_txt(path, is_pred=False):
    """
    Returns list of dicts:
      GT:  {'cls': int, 'box': xyxy}
      Pred:{'cls': int, 'box': xyxy, 'conf': float}
    Accepts lines:
      cls xc yc w h
      cls xc yc w h conf
    """
    items = []
    if not os.path.isfile(path):
        return items

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            box = yolo_xywh_to_xyxy((xc, yc, w, h))

            if is_pred:
                conf = 1.0
                if len(parts) >= 6:
                    try:
                        conf = float(parts[5])
                    except ValueError:
                        conf = 1.0
                items.append({"cls": cls, "box": box, "conf": conf})
            else:
                items.append({"cls": cls, "box": box})

    return items


def build_index(gt_dir, pred_dir):
    """Collect all basenames (without extension) from both dirs"""
    gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))
    pr_files = glob.glob(os.path.join(pred_dir, "*.txt"))

    basenames = set()
    for p in gt_files + pr_files:
        basenames.add(os.path.splitext(os.path.basename(p))[0])

    return sorted(list(basenames))


def compute_ap_101(recalls, precisions):
    """
    COCO-style 101-point interpolated AP:
    AP = mean_{r in {0,0.01,...,1}} max_{recall>=r} precision
    """
    if len(recalls) == 0:
        return 0.0
    recalls = np.array(recalls, dtype=np.float32)
    precisions = np.array(precisions, dtype=np.float32)

    ap = 0.0
    for r in np.linspace(0, 1, 101):
        p = precisions[recalls >= r].max() if np.any(recalls >= r) else 0.0
        ap += p
    return float(ap / 101.0)


def evaluate(gt_dir, pred_dir, iou_thr=0.5):
    basenames = build_index(gt_dir, pred_dir)

    # Per image parsed data
    gt_by_img = {}
    pr_by_img = {}

    all_classes = set()

    for bn in basenames:
        gt_path = os.path.join(gt_dir, bn + ".txt")
        pr_path = os.path.join(pred_dir, bn + ".txt")

        gt = read_yolo_txt(gt_path, is_pred=False)
        pr = read_yolo_txt(pr_path, is_pred=True)

        gt_by_img[bn] = gt
        pr_by_img[bn] = pr

        for g in gt:
            all_classes.add(g["cls"])
        for p in pr:
            all_classes.add(p["cls"])

    if not basenames:
        raise RuntimeError("Δεν βρέθηκαν .txt αρχεία στους φακέλους που δώσατε.")

    if not all_classes:
        # no GT, no preds
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "mAP50": 0.0, "ap_per_class": {}
        }

    classes = sorted(list(all_classes))

    ap_per_class = {}
    total_gt = 0
    total_tp = 0
    total_fp = 0

    # Evaluate per class
    for c in classes:
        # Ground truth per image for this class
        gt_c = {}
        npos = 0
        for img_id in basenames:
            gts = [g["box"] for g in gt_by_img[img_id] if g["cls"] == c]
            if gts:
                gt_c[img_id] = gts
                npos += len(gts)

        # Predictions list for this class across images
        preds = []
        for img_id in basenames:
            for p in pr_by_img[img_id]:
                if p["cls"] == c:
                    preds.append((p["conf"], img_id, p["box"]))

        preds.sort(key=lambda x: x[0], reverse=True)

        # Bookkeeping for matching: for each image, track which gt boxes are already matched
        matched = {img_id: np.zeros(len(gt_c.get(img_id, [])), dtype=bool) for img_id in basenames}

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for i, (conf, img_id, pbox) in enumerate(preds):
            gts = gt_c.get(img_id, [])
            if not gts:
                fp[i] = 1.0
                continue

            # Find best IoU among unmatched gts
            best_iou = -1.0
            best_j = -1
            for j, gtbox in enumerate(gts):
                if matched[img_id][j]:
                    continue
                iou = iou_xyxy(pbox, gtbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_thr and best_j >= 0:
                tp[i] = 1.0
                matched[img_id][best_j] = True
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        if npos > 0:
            recalls = tp_cum / float(npos)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            ap = compute_ap_101(recalls, precisions)
        else:
            # No GT for this class -> AP undefined; set 0
            ap = 0.0

        ap_per_class[c] = ap

        # For global micro P/R/F1: only count classes that have GT,
        # but it’s common to count all preds too. Here we:
        # - total_gt counts all GT boxes (all classes)
        # - total_tp counts matched TPs (all classes)
        # - total_fp counts FPs (all classes)
        total_gt += npos
        total_tp += int(tp.sum())
        total_fp += int(fp.sum())

    fn = max(0, total_gt - total_tp)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # mAP50: average AP over classes with at least 1 GT (common practice)
    classes_with_gt = [c for c in classes if any(g["cls"] == c for img in basenames for g in gt_by_img[img])]
    if classes_with_gt:
        mAP50 = float(np.mean([ap_per_class[c] for c in classes_with_gt]))
    else:
        mAP50 = 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mAP50": float(mAP50),
        "ap_per_class": ap_per_class,
        "total_gt": int(total_gt),
        "total_tp": int(total_tp),
        "total_fp": int(total_fp),
        "total_fn": int(fn),
        "classes": classes,
        "classes_with_gt": classes_with_gt,
        "num_images": len(basenames),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", default="gt_labels", help="Folder with ground-truth YOLO txt labels (gt_labels)")###################################################
    ap.add_argument("--pred_dir", default="predicted_labels", help="Folder with predicted YOLO txt labels (raspi_labels)")###################################################
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default 0.5)")
    ap.add_argument("--per_class", action="store_true", help="Print AP per class")
    args = ap.parse_args()

    res = evaluate(args.gt_dir, args.pred_dir, iou_thr=args.iou)

    print("\n=== YOLO TXT Evaluation (IoU={:.2f}) ===".format(args.iou))
    print("Images:      ", res["num_images"])
    print("Total GT:    ", res["total_gt"])
    print("Total TP:    ", res["total_tp"])
    print("Total FP:    ", res["total_fp"])
    print("Total FN:    ", res["total_fn"])
    print("--------------------------------------")
    print("Precision:   {:.6f}".format(res["precision"]))
    print("Recall:      {:.6f}".format(res["recall"]))
    print("F1:          {:.6f}".format(res["f1"]))
    print("mAP50:       {:.6f}".format(res["mAP50"]))

    if args.per_class:
        print("\nAP50 per class:")
        for c in sorted(res["ap_per_class"].keys()):
            tag = " (no GT)" if c not in res["classes_with_gt"] else ""
            print("  class {:>3d}: {:.6f}{}".format(c, res["ap_per_class"][c], tag))


if __name__ == "__main__":
    main()
