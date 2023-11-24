import os
import cv2
import numpy as np
import config
import load_data
from tqdm import tqdm


def label_view(img_path, label_path: str, save_path, mode, vis_classes, save_crops=True, save_empty=True,
               label_hide=False):
    assert os.path.exists(img_path), "the img_path do not exist!"
    assert os.path.exists(label_path), "the label_path do not exist!"

    os.makedirs(os.path.join(save_path, 'label'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'error'), exist_ok=True)

    if save_empty:
        os.makedirs(os.path.join(save_path, 'empty'), exist_ok=True)
    if save_crops:
        for i, cl in enumerate(vis_classes):
            os.makedirs(os.path.join(save_path, 'crops', cl), exist_ok=True)

    modeInVisualize = {'LabelView_COCO': "COCO", 'LabelView_VOC': "VOC", 'LabelView_YOLO': "YOLO"}
    show_mode = modeInVisualize[mode]

    # assert
    if show_mode == "COCO":
        assert label_path.split('.')[-1].lower() == 'json', "the label path is wrong! not json file!"
    elif show_mode == "VOC":
        assert os.path.isdir(label_path), "the label path is wrong! not a directory!"
        for file in os.listdir(label_path):
            assert file.split('.')[-1].lower() == 'xml', "the label path is wrong! not VOC file!"
    else:
        assert os.path.isdir(label_path), "the label path is wrong! not a directory!"
        for file in os.listdir(label_path):
            assert file.split('.')[-1].lower() == 'txt', "the label path is wrong! not YOLO file!"

    imgsList = [img for img in os.listdir(img_path) if os.path.splitext(img)[-1].lower() in config.IMG_FORMATS]
    imgsSet = set(imgsList)

    # 读取数据，返回 json 格式
    jsonInfo, imgId2Name = getattr(load_data, 'load' + show_mode)(config.Classes, img_path, label_path)

    # with open(os.path.join(save_path, 'xx.json'), 'w') as f:
    #     json.dump(jsonInfo, f)

    # color setting
    clsId2Colors = [(COLORS[i + 17] * 255 * 0.7).astype(np.float32).tolist() for i in range(len(vis_classes))]

    # 遍历每个标注信息
    for anno in tqdm(jsonInfo['annotations'], desc='visualizing...'):
        image_id, bbox, category_id = anno['image_id'], anno['bbox'], anno['category_id']

        if imgId2Name[image_id] not in imgsList or jsonInfo['categories'][category_id - 1]['name'] not in vis_classes:
            continue
        imgsSet.discard(imgId2Name[image_id])

        tar_path = os.path.join(save_path, os.path.join(save_path, 'label', imgId2Name[image_id]))
        if os.path.exists(tar_path):
            img = cv2.imread(tar_path)
        else:
            img = cv2.imread(os.path.join(img_path, imgId2Name[image_id]))

        left_top, right_down = (int(bbox[0]), int(bbox[1])), (int(bbox[2]) + int(bbox[0]), int(bbox[3]) + int(bbox[1]))
        color = clsId2Colors[category_id - 1]
        imgH = img.shape[0]

        # label text
        text_size = max(imgH / 500 * 0.3, 0.5)
        text_thinkness = max(int(imgH / 500 * 1), 1)
        rectangle_thinkness = text_thinkness

        # 执行画框等一系列操作
        try:
            img_new = cv2.rectangle(img, left_top, right_down, color, thickness=rectangle_thinkness)
            if not label_hide:
                # 在框的上方写标签
                text_loc = (int(bbox[0]), int(bbox[1]) - 5)
                cv2.putText(img_new, jsonInfo['categories'][category_id - 1]['name'], text_loc,
                            cv2.FONT_HERSHEY_SIMPLEX, text_size, color, text_thinkness)
            if save_crops:
                img_origin = cv2.imread(os.path.join(img_path, imgId2Name[image_id]))
                img_crop = img_origin[int(bbox[1]): int(bbox[1]) + int(bbox[3]), int(bbox[0]):int(bbox[0]) + int(bbox[2]), :]

                i = 1
                while True:
                    output_path = os.path.join(save_path, 'crops', jsonInfo['categories'][category_id - 1]['name'],
                                               '{}_crop_{}.{}'.format(imgId2Name[image_id].split('.')[0], i,
                                                                      imgId2Name[image_id].split('.')[-1]))
                    i += 1
                    if not os.path.exists(output_path):
                        break

                cv2.imwrite(output_path, img_crop)

            cv2.imwrite(os.path.join(save_path, os.path.join(save_path, 'label', imgId2Name[image_id])), img_new)
        except Exception as e:
            # print(e)
            cv2.imwrite(os.path.join(save_path, os.path.join(save_path, 'error', imgId2Name[image_id])), img)

        # 带有分割信息
        if anno['segmentation']:
            os.makedirs(os.path.join(save_path, 'mask'), exist_ok=True)
            tar_path = os.path.join(save_path, os.path.join(save_path, 'mask', imgId2Name[image_id]))
            if os.path.exists(tar_path):
                img = cv2.imread(tar_path)
            else:
                img = cv2.imread(os.path.join(img_path, imgId2Name[image_id]))

            try:
                cv2.polylines(img, [np.array(anno['segmentation'], dtype=np.int).reshape(-1, 2)], True, color, 1)
                cv2.imwrite(tar_path, img)
            except Exception as e:
                pass

    if save_empty:
        import shutil
        for imgName in imgsSet:
            # empty label
            shutil.copy(os.path.join(img_path, imgName), os.path.join(save_path, 'empty', imgName))


# yolo color
COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,  # 17 yellow
        0.667, 0.333, 0.000,  # 18 brown
        0.667, 0.667, 0.000,  # 19 olive
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
