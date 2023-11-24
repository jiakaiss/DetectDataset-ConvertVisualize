from utils import *
import json
import os
import cv2
from tqdm import tqdm


def COCO2COCO(jsonInfo: dict, savePath, imagesPath, imgId2Name):
    os.makedirs(os.path.join(savePath, 'labels'), exist_ok=True)
    Json_path = os.path.join(savePath, 'labels', 'annotations.json')
    with open(Json_path, 'w') as f:
        json.dump(jsonInfo, f)


def COCO2YOLO(jsonInfo, savePath, imagesPath, imgId2Name: dict):
    os.makedirs(os.path.join(savePath, 'labels'), exist_ok=True)

    for ann in tqdm(jsonInfo['annotations'], desc='traversing json file, COCO2YOLO...'):
        try:
            imgName = imgId2Name[ann['image_id']]
        except Exception as e:
            print(ann['image_id'])
            break

        txtName = os.path.join(savePath, 'labels', imgName.split('.')[0] + '.txt')

        # 标签为空时保存空txt
        if ann['bbox'] is None:
            file = open(txtName, 'a')
            file.close()
            continue

        bbox = ann['bbox']
        img = cv2.imread(os.path.join(imagesPath, imgName))
        imgH, imgW = img.shape[0], img.shape[1]

        # 若存在分割信息，优先保存分割信息
        if ann['segmentation'] and ann['segmentation'] != []:
            if not isinstance(ann['segmentation'], list):
                continue

            try:
                segments = ann['segmentation'][0]
                # print('segments', len(segments))
                with open(txtName, 'a') as file:
                    # -1, yolo格式
                    file.write(str(ann['category_id'] - 1))
                    for idx in range(0, len(segments), 2):
                        file.write(' ' + str(segments[idx] / imgW) + ' ' + str(segments[idx + 1] / imgH))
                    file.write('\n')
            except Exception as e:
                print(e)

        else:
            yolo_xywh = coco2yolo(bbox, imgW, imgH)
            with open(txtName, 'a') as file:
                # -1， yolo格式
                label = str(ann['category_id'] - 1) + ' ' + yolo_xywh + '\n'
                file.write(label)


def COCO2VOC(jsonInfo, savePath, imagesPath, imgId2Name: dict):
    os.makedirs(os.path.join(savePath, 'labels'), exist_ok=True)

    cateId2Name = {}
    for cate in jsonInfo['categories']:
        cateId2Name[cate['id']] = cate['name']
    # print('cateId2Name=', len(cateId2Name))

    for imgInfo in jsonInfo['images']:
        imgName = imgInfo['file_name']
        name, extension = os.path.splitext(imgName)

        img = cv2.imread(os.path.join(imagesPath, imgName))
        imgH, imgW, imgDepth = img.shape
        xmlSavePath = os.path.join(savePath, 'labels', name + '.xml')
        tree = generateImgInfo(imagesPath, imgName, imgH, imgW, imgDepth)
        tree.write(xmlSavePath, encoding='utf-8', xml_declaration=True, method='xml')

    # write object info
    for ann in tqdm(jsonInfo['annotations'], desc='traversing json file, COCO2VOC...'):
        imgName = imgId2Name[ann['image_id']]
        name, extension = os.path.splitext(imgName)
        xmlName = os.path.join(savePath, 'labels', name + '.xml')

        # 空标签处理
        if ann['bbox'] is None:
            continue

        bbox = ann['bbox']
        bbox = [int(box) for box in bbox]

        cateName = cateId2Name[ann['category_id']]

        assert os.path.exists(xmlName)

        tree = ET.parse(xmlName)
        tree = insertObject(tree, cateName, bbox)
        tree.write(xmlName, encoding='utf-8', xml_declaration=True, method='xml')

