import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_hub as hub

from eval.extract_vp_utils import filter_boxes_bcp
from models.hourglass import load_model, parse_command_line
from utils.diamond_space import get_focal, process_heatmaps
from utils.video import get_cap
from pathlib import Path
from typing import Tuple
import copy

import pdb

def pretty_line(img: np.array, p1: Tuple[float], p2: Tuple[float], color: Tuple[int, int, int],
                thickness: int) -> np.array:
    f2int = lambda p: (int(p[0]), int(p[1]))
    img = cv2.line(img, f2int(p1), f2int(p2), (0, 0, 0), thickness + 5, cv2.LINE_AA)
    img = cv2.line(img, f2int(p1), f2int(p2), color, thickness, cv2.LINE_AA)
    return img


def write_helper_on_image(input_frame):
    texts = ['1. Press Enter to re-compute VPs', '2. Press S to save the VPs']
    start_x = 10
    start_y = 30
    step_y = 30
    for text in texts:
        cv2.putText(input_frame, text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(input_frame, text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 2,
                    cv2.LINE_AA)
        start_y += step_y
    return input_frame


def preview():
    args = parse_command_line()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    heatmap_model, scales, _, _ = load_model(args)
    print("Heatmap model loaded!")

    # Load TensorFlow 1.x model
    detector = hub.Module("https://www.kaggle.com/models/google/faster-rcnn-inception-resnet-v2/TensorFlow1/faster-rcnn-openimages-v4-inception-resnet-v2/1")
    input_tensor = tf.placeholder(tf.float32, [1, None, None, 3])
    output_dict = detector(input_tensor, as_dict=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print("Object detection model loaded!")

    cap = get_cap(args.path)

    vp1s, vp2s, fs, ms, b1s, b2s = [], [], [], [], [], []
    prev_edge = None
    ret = True
    outer_counter = 0

    while ret:
        # pdb.set_trace()
        ret, frame = cap.read()
        if not ret:
            break

        outer_counter += 1

        # Preprocess frame for detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = frame_rgb[np.newaxis, ...].astype(np.float32) / 255.0

        result = sess.run(output_dict, feed_dict={input_tensor: input_frame})
        # pdb.set_trace()
        boxes = result["detection_boxes"]
        class_names = result["detection_class_names"]
        class_entities = result["detection_class_entities"]
        scores = result["detection_scores"]

        # Filter for 'Car'
        car_filter = np.logical_and(scores > 0.5, np.array(class_entities) == b'Car')
        boxes = boxes[car_filter]
        scores = scores[car_filter]

        boxes, scores, _, prev_edge = filter_boxes_bcp(boxes, scores, frame, prev_edge)

        showing_frame = copy.deepcopy(frame)
        showing_frame = cv2.resize(showing_frame, args.resize_imshow_frame_into)
        # showing_frame = write_helper_on_image(showing_frame)
        # cv2.imshow("Frame", showing_frame)
        cv2.imwrite(f"image_outputs/Frame{outer_counter}.png", showing_frame)
        print("imwrite 1")
        cv2.waitKey(30)

        # pdb.set_trace()
        inner_counter = 0
        for box in boxes:
            x_min = int(1920 * box[1]) # todo: remove magic numbers
            y_min = int(1080 * box[0])
            x_max = int(1920 * box[3] + 1)
            y_max = int(1080 * box[2] + 1)

            box_center = np.array([x_min + x_max, y_min + y_max]) / 2
            box_scale = np.array([x_max - x_min, y_max - y_min]) / 2

            car = frame[y_min:y_max, x_min:x_max, :]
            car = cv2.resize(car, (args.input_size, args.input_size), cv2.INTER_CUBIC)

            heatmap_pred = heatmap_model.predict(car[np.newaxis, ...] / 255.0)
            pred_vps, pred_vars = process_heatmaps(heatmap_pred[-1], scales)

            vp1_var = pred_vars[0, :, 0]
            vp2_var = pred_vars[0, :, 1]
            vp1_var_idx = np.argmin(vp1_var, axis=-1)
            vp2_var_idx = np.argmin(vp2_var, axis=-1)

            vp1_box = pred_vps[0, vp1_var_idx, :2]
            vp2_box = pred_vps[0, vp2_var_idx, 2:]

            vp1 = box_scale * vp1_box + box_center
            vp2 = box_scale * vp2_box + box_center

            principal_point = np.array([frame.shape[1] / 2 + 0.5, frame.shape[0] / 2 + 0.5])
            focal = get_focal(vp1, vp2, principal_point)
            m = (vp1[1] - vp2[1]) / (vp1[0] - vp2[0])
            b1 = vp1[1] - m * vp1[0]
            b2 = vp2[1] - m * vp2[0]

            inner_counter += 1

            if not np.isnan(focal) and not np.isinf(m) and not np.isnan(m):
                vp1s.append(vp1)
                vp2s.append(vp2)
                fs.append(focal)
                ms.append(m)
                b1s.append(b1)
                b2s.append(b2)

                print("VP1: {} \t VP2: {} \t focal: {}".format(vp1, vp2, focal))
                print("Median horizon y = {} * x + {}".format(np.nanmedian(ms), np.nanmedian(np.concatenate([b1s, b2s]))))
                print("Median focal {}".format(np.nanmedian(fs)))

                frame_scale = np.copy(frame)
                try:
                    frame_scale = pretty_line(frame_scale, box_center, vp1, (0, 255, 0), 2)
                    frame_scale = pretty_line(frame_scale, box_center, vp2, (0, 0, 255), 2)
                    frame_scale = pretty_line(frame_scale, vp1, vp2, (0, 255, 255), 2)
                except Exception:
                    pass

                frame_scale = cv2.resize(frame_scale, args.resize_imshow_frame_into)
                black_image = np.zeros((frame_scale.shape[0], args.input_size, 3), np.uint8)
                black_image[0:args.input_size, 0:args.input_size, :] = car
                frame_scale = cv2.hconcat((frame_scale, black_image))
                # frame_scale = write_helper_on_image(frame_scale)

                # cv2.imshow("Frame", frame_scale)
                cv2.imwrite(f"image_outputs/Frame{outer_counter}_{inner_counter}.png", frame_scale)
                print(f"imwrite {inner_counter}")
                # if cv2.waitKey(0) == ord('s'):
                # save heatmaps

                print("VP1 var idx: ", vp1_var_idx)
                print("VP2 var idx: ", vp2_var_idx)
                print("VP1 var: ", vp1_var)
                print("VP2 var: ", vp2_var)

                print("xmin: ", x_min)
                print("xmax: ", x_max)
                print("ymin: ", y_min)
                print("ymax: ", x_max)

                Path("datasets/vis/").mkdir(parents=True, exist_ok=True)
                cv2.imwrite("datasets/vis/frame_preview.png", frame)
                cv2.imwrite("datasets/vis/car_preview.png", car)

                heatmap_pred[-1][heatmap_pred[-1] < 0] = 0
                for vp_idx in range(2):
                    for scale_idx, scale in enumerate(scales):
                        idx = len(scales) * vp_idx + scale_idx
                        heatmap = heatmap_pred[-1][0, :, :, idx]
                        heatmap_img = np.uint8(255 * heatmap / np.max(heatmap))
                        heatmap_colored = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_PARULA)
                        cv2.imwrite(f"datasets/vis/heatmap_preview_vp{vp_idx+1}_s{scale}.png", heatmap_colored)


if __name__ == '__main__':
    preview()