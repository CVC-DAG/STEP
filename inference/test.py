# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import multiprocessing as mp
import time
import tqdm
import json
import glob
import re

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import torch
import numpy as np
import cv2
import editdistance

from predictor import VisualizationDemo
from adet.config import get_cfg
from adet.utils.queries import generate_query
from adet.utils.queries import ind_to_chr, indices_to_text

# constants
WINDOW_NAME = "COCO detections"

opj = os.path.join
classes = ["bic", "phone", "tare", "uic", "tonnage", "lp", "weights"]


def build_label_queries():
    label_queries = dict()

    bic_query = generate_query("llllsnnnnnnsn", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    label_queries["bic"] = [bic_query]

    phone_query = torch.clamp(generate_query("nnnsnnnsnnnn", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) +
                             generate_query("nnn-nnn-nnnn", [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]) +
                             generate_query("nnn.nnn.nnnn", [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]), max=1.0)
    label_queries["phone"] = [phone_query]

    tare_query1 = torch.clamp(generate_query("nnsnnnkg", [0, 0, 0, 0, 0, 0, 1, 1]) +
                             generate_query("nn.nnnKG", [0, 0, 1, 0, 0, 0, 1, 1]) +
                             generate_query("nn,nnnkg", [0, 0, 1, 0, 0, 0, 1, 1]), max=1.0)
    tare_query2 = torch.clamp(generate_query("nnnnnkg", [0, 0, 0, 0, 0, 1, 1]) +
                             generate_query("nnnnnKG", [0, 0, 0, 0, 0, 1, 1]), max=1.0)
    label_queries["tare"] = [tare_query1, tare_query2]

    uic_query1 = generate_query("nnnnnnnnnnn-n", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    uic_query2 = generate_query("nnsnnsnnnnsnnn-n", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    uic_query3 = generate_query("nnsnnsnnnsnsnnn-n", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    label_queries["uic"] = [uic_query3, uic_query2, uic_query1]

    tonage_query1 = generate_query("nnt", [0, 0, 1])
    tonage_query2 = generate_query("nn.nt", [0, 0, 1, 0, 1])
    label_queries["tonnage"] = [tonage_query2, tonage_query1]

    lp_query = generate_query("lllsnnnn", [0, 0, 0, 0, 0, 0, 0, 0])
    label_queries["lp"] = [lp_query]

    return label_queries


def nms(bounding_boxes, confidence_score, recogs, threshold):
    if len(bounding_boxes) == 0:
        return [], []

    boxes = np.array(bounding_boxes)
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]
    score = np.array(confidence_score)

    picked_boxes = []
    picked_score = []
    picked_recogs = []

    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    order = np.argsort(score)

    while order.size > 0:
        index = order[-1]

        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_recogs.append(recogs[index])

        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score, picked_recogs


def bb_intersection_over_union(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def filter_preds(boxes, recs, confs):
    matched_boxes = []
    matched_scores = []
    matched_recs = []
    for pred_box, pred_rec, rec_conf in zip(boxes, recs, confs):
        if rec_conf > args.rc:
            matched_boxes.append(pred_box)
            matched_scores.append(rec_conf)
            matched_recs.append(pred_rec)

    if matched_boxes:
        matched_boxes, matched_scores, matched_recs = nms(matched_boxes, matched_scores, matched_recs, 0.5)
    matches = []
    for box, rec, rec_score in zip(matched_boxes, matched_recs, matched_scores):
        matches.append((box, rec, rec_score))

    return matches


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "-rc",
        type=float,
        default=0.6,
        help="Recognition min. confidence",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--vis",
        help="visualization",
        action='store_true'
    )

    parser.add_argument(
        "--det",
        help="eval detection",
        action="store_true"
    )

    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    gt_path = "datasets/structured_test"
    gt_files = glob.glob(opj(gt_path, "*.json"))

    class_metrics = dict()
    label_queries = build_label_queries()
    edit_dists = []

    for gt_file in tqdm.tqdm(gt_files):
        gt = json.load(open(gt_file))
        img = read_image(opj(gt_path, gt["imagePath"]), format="BGR")

        label_to_shapes = dict()
        for shape in gt["shapes"]:
            label = shape["label"]
            if shape["label"] not in label_to_shapes.keys():
                label_to_shapes[label] = [shape]
            else:
                label_to_shapes[label].append(shape)

        all_gts = []
        all_matches = []
        for label in label_to_shapes.keys():
            if label == "weights":  # ignore weights class for now
                continue
            if label not in class_metrics.keys():
                class_metrics[label] = {"TP": 0, "FP": 0, "FN": 0}

            queries = label_queries[label]
            pred_boxes = []
            pred_recs = []
            rec_confs = []
            for query in queries:
                query_input = [query.unsqueeze(dim=0).type(torch.FloatTensor).cuda()]

                start_time = time.time()

                predictions, visualized_output = demo.run_on_image(img, query_input)

                polygons = predictions["instances"].get("polygons")
                for pol_num in range(polygons.shape[0]):
                    xs = polygons[pol_num, ::2]
                    ys = polygons[pol_num, 1::2]
                    bbox = [float(min(xs)),
                            float(min(ys)),
                            float(max(xs)),
                            float(max(ys))]
                    pred_boxes.append(bbox)

                recs = predictions["instances"].get("recs")
                pred_recs.extend([list(map(lambda x: int(x), list(recs[i])))
                                  for i in range(recs.shape[0])])

                if len(recs):
                    rec_confs.extend([float(v.item()) for v in
                                      predictions["instances"].get("rec_scores").max(dim=2)[0].min(dim=1)[0]])

            matches = filter_preds(pred_boxes, pred_recs, rec_confs)
            all_matches.extend(matches)

            TP = 0
            for shape in label_to_shapes[label]:
                points = np.array([c for p in shape["points"] for c in p])
                xs = points[::2]
                ys = points[1::2]
                gt_bbox = [float(min(xs)),
                           float(min(ys)),
                           float(max(xs)),
                           float(max(ys))]
                gt_trans = re.sub(r'[^\w]', '', shape["transcription"].lower())
                all_gts.append([gt_bbox, gt_trans])

                for pred_box, pred_rec, rec_conf in matches:
                    iou = bb_intersection_over_union(gt_bbox, pred_box)
                    pred_rec = re.sub(r'[^\w]', '', indices_to_text(pred_rec).lower())
                    if iou > 0.5:
                        edit_dists.append(editdistance.eval(gt_trans, pred_rec))
                    if iou > 0.5 and (gt_trans == pred_rec or args.det):
                        TP += 1
                        break

            FP = len(matches) - TP
            FN = len(label_to_shapes[label]) - TP

            class_metrics[label]["TP"] += TP
            class_metrics[label]["FP"] += FP
            class_metrics[label]["FN"] += FN

        if args.vis:
            import matplotlib.pyplot as plt
            import matplotlib.patches as pat

            fig, ax = plt.subplots()

            for gt_bbox, gt_trans in all_gts:
                rect = pat.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1],
                                     linewidth=3, edgecolor='g', facecolor='none')
                plt.text(gt_bbox[2], gt_bbox[1], gt_trans, color="g")
                ax.add_patch(rect)

            for pred_box, pred_rec, rec_conf in all_matches:  # zip preds to see all the preds
                rect = pat.Rectangle((pred_box[0], pred_box[1]), pred_box[2] - pred_box[0],
                                     pred_box[3] - pred_box[1], linewidth=1,
                                     edgecolor='r', facecolor='none')
                plt.text(pred_box[0], pred_box[1],
                         "{:.2f} - ".format(rec_conf) + "".join(ind_to_chr[c] if c != 96 else "" for c in pred_rec), color="r")
                ax.add_patch(rect)

            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()

    for label in class_metrics.keys():
        p = class_metrics[label]["TP"] / (class_metrics[label]["TP"] + class_metrics[label]["FP"])
        r = class_metrics[label]["TP"] / (class_metrics[label]["TP"] + class_metrics[label]["FN"])
        print(label, "prec:", p, "rec:", r, " Fsc:", 2 * ((p * r) / (p + r)), "ed:", np.array(edit_dists).mean())

