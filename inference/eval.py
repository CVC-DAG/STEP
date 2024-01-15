# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import argparse
import multiprocessing as mp
import time
import tqdm
import json

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import torch
import numpy as np
import cv2
import editdistance

from predictor import VisualizationDemo
from adet.config import get_cfg
from adet.utils.queries import text_to_query_t1, ind_to_chr, indices_to_text, compare_queries

# constants
WINDOW_NAME = "COCO detections"


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

    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    gt_path = "datasets/hiertext/validation.jsonl"
    base_path = "datasets/hiertext/validation/"
    gt = json.load(open(gt_path))
    imgid_to_obj = {}

    for img_info in gt["images"]:
        imgid_to_obj[img_info["id"]] = img_info
        img_info["annotations"] = []

    for ann in gt["annotations"]:
        imgid_to_obj[ann["image_id"]]["annotations"].append(ann)

    precs = []
    recs = []
    edit_dists = []

    TPs = 0
    FPs = 0
    FNs = 0

    ims = gt["images"]

    for img_info in tqdm.tqdm(ims):
        # use PIL, to be consistent with evaluation
        img_path = base_path + img_info["file_name"]

        if not len(img_info["annotations"]):
            continue

        queries_to_anns = {}
        for ann in img_info["annotations"]:
            query = text_to_query_t1(ann["rec"])
            query_hash = hash(tuple(tuple(x) for x in query.tolist()))  # stupid way to get a -proper- tensor hash
            if query_hash in queries_to_anns.keys():
                queries_to_anns[query_hash][0].append(ann)
            else:
                queries_to_anns[query_hash] = ([ann], query)
        assert len([ann for k in queries_to_anns.keys() for ann in queries_to_anns[k][0]]) \
               == len(img_info["annotations"])

        for query_hash in queries_to_anns.keys():
            accepted, query = queries_to_anns[query_hash]
            img = read_image(img_path, format="BGR")
            query_input = [query.unsqueeze(dim=0).type(torch.FloatTensor).cuda()]

            start_time = time.time()

            predictions, visualized_output = demo.run_on_image(img, query_input)

            polygons = predictions["instances"].get("polygons")
            pred_boxes = []
            for pol_num in range(polygons.shape[0]):
                xs = polygons[pol_num, ::2]
                ys = polygons[pol_num, 1::2]
                bbox = [float(min(xs)),
                        float(min(ys)),
                        float(max(xs)),
                        float(max(ys))]
                pred_boxes.append(bbox)

            pred_recs = predictions["instances"].get("recs")
            pred_recs = [list(map(lambda x: int(x), list(pred_recs[i]))) for i in range(pred_recs.shape[0])]
            if len(pred_recs):
                rec_confs = [float(v.item()) for v in
                             predictions["instances"].get("rec_scores").max(dim=2)[0].min(dim=1)[0]]
            else:
                rec_confs = []

            matched_boxes = []
            matched_scores = []
            matched_recs = []
            for pred_box, pred_rec, rec_conf in zip(pred_boxes, pred_recs, rec_confs):
                if compare_queries(text_to_query_t1(pred_rec), query) and rec_conf > args.rc:
                    matched_boxes.append(pred_box)
                    matched_scores.append(rec_conf)
                    matched_recs.append(pred_rec)

            if matched_boxes:
                matched_boxes, matched_scores, matched_recs = nms(matched_boxes, matched_scores, matched_recs, 0.75)
            matches = []
            for box, rec in zip(matched_boxes, matched_recs):
                matches.append((box, rec))

            TP = 0
            for gt_obj in accepted:
                gt_box = gt_obj["bbox"]
                gt_box = [gt_box[0], gt_box[1],
                           gt_box[0] + gt_box[2],
                           gt_box[1] + gt_box[3]]
                for pred_box, pred_rec in matches:
                    iou = bb_intersection_over_union(gt_box, pred_box)
                    if iou > 0.5:
                        edit_dists.append(editdistance.eval(gt_obj["rec"], pred_rec))
                    if iou > 0.5 and gt_obj["rec"] == pred_rec:
                        TP += 1
                        break
            FP = len(matches) - TP
            FN = len(accepted) - TP

            TPs += TP
            FPs += FP
            FNs += FN
            if TP + FP != 0:
                precs.append(TP / (TP + FP))

            if TP + FN != 0:
                recs.append(TP / (TP + FN))

            if args.vis:  # and hit_space:
                import matplotlib.pyplot as plt
                import matplotlib.patches as pat

                fig, ax = plt.subplots()

                for gt_obj in accepted:
                    gt_box = gt_obj["bbox"]
                    gt_text = indices_to_text(gt_obj["rec"])
                    rect = pat.Rectangle((gt_box[0], gt_box[1]), gt_box[2], gt_box[3], linewidth=3,
                                             edgecolor='g', facecolor='none')
                    plt.text(gt_box[0]+gt_box[2], gt_box[1], gt_text, color="g")
                    ax.add_patch(rect)

                for pred_box, pred_rec in matches: # zip preds to see all the preds
                    rect = pat.Rectangle((pred_box[0], pred_box[1]), pred_box[2] - pred_box[0],
                                         pred_box[3] - pred_box[1], linewidth=1,
                                         edgecolor='r', facecolor='none')
                    plt.text(pred_box[0], pred_box[1],
                             "".join(ind_to_chr[c] if c != 96 else "" for c in pred_rec), color="r")
                    ax.add_patch(rect)

                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show()

        # logger.info(
        #     "{}: detected {} instances in {:.2f}s".format(
        #         img_path, len(predictions["instances"]), time.time() - start_time
        #     )
        # )

    print("overall prec:", TPs / (TPs + FPs) if TPs + FPs else None, " overall rec:", TPs / (TPs + FNs),
          "overall ed:", np.array(edit_dists).mean())