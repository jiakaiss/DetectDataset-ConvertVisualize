import cv2
import os
import json
import shutil
from tqdm import tqdm
from utils import *

"""
将所有数据读取并统一转换为coco形式
{
    "info": info,                 //包含整个数据集信息, 不重要
    "licenses": [license],        //许可证类别，不重要
    "images": [image],            //list,重要
    "annotations": [annotation],  //list,重要
    "categories": [category]      //list,分类类别信息，重要
} 
"""

# include image suffixes
IMG_FORMATS = {'.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pgm'}


def loadVOC(classNames: dict, imagesPath: str, xmlPath: str):

    Annotations = {"info": {}, "licenses": [], "images": [], "annotations": [], "categories": []}

    classNames_list = list(classNames.keys())

    # categories
    for i, className in enumerate(classNames_list):
        Annotations["categories"].append({"id": i + 1, "name": className})

    imgNameList = [f for f in os.listdir(imagesPath) if os.path.splitext(f)[1].lower() in IMG_FORMATS]

    imgIdIdx = 0
    imgId2Name, imgName2Id, name2ImgName = {}, {}, {}
    for imgName in imgNameList:
        imgId2Name[imgIdIdx] = imgName
        imgName2Id[imgName] = imgIdIdx
        name = os.path.splitext(imgName)[0]
        name2ImgName[name] = imgName
        imgIdIdx += 1

    cateId2Name, cateName2Id = {}, {}
    for i, cateName in enumerate(classNames_list):
        cateId2Name[i + 1] = cateName  # coco class start from 1
        cateName2Id[cateName] = i + 1

    xmlList = []
    for root, _, files in os.walk(xmlPath):
        # xmlList = files
        for file in files:
            if os.path.splitext(file)[1] == '.xml':
                xmlList.append(file)

    annId = 0
    Annotations_mask = {}  # keys为img_id, values为bbox组成的二维数组
    for xmlFile in xmlList:
        name = xmlFile[:-4]
        imgName = name2ImgName[name]
        imgH, imgW, _ = cv2.imread(os.path.join(imagesPath, imgName)).shape

        # 写入image信息
        Annotations["images"].append(
            {
                "file_name": imgName,
                "height": imgH,
                "width": imgW,
                "id": imgName2Id[imgName]
            })

        tree = ET.parse(os.path.join(xmlPath, xmlFile))
        root = tree.getroot()

        count = 0  # 用于统计单个图片中bbox个数, 为空时保证有标签文件输出
        
        # 逐个bbox进行处理
        for object in root.findall('object'):
                
            labelName = object.find('name').text

            try:
                xmin = float(object.findall('xmin').text)
                ymin = float(object.findall('ymin').text)
                xmax = float(object.findall('xmax').text)
                ymax = float(object.findall('ymax').text)
                xmin = 0 if xmin < 0 else xmin
                x, y, w, h = coco_xywh = voc2coco([xmin, ymin, xmax, ymax])
            except:
                xyxy = [float(it.text) for it in object.find('bndbox')]
                x, y, w, h = coco_xywh = voc2coco(xyxy)

            classId = cateName2Id[labelName]
            # 写入annotations
            Annotations["annotations"].append(
                {
                    "segmentation": [],
                    "iscrowd": 0,  # 0 or 1
                    "area": w * h,  # float or double
                    "image_id": imgName2Id[imgName],  # int
                    "bbox": coco_xywh,  # list[float], [x,y,w,h]
                    "category_id": classId,
                    "id": annId  # int
                }
            )
            annId += 1
            count += 1

        # 空标签处理
        if count == 0:
            Annotations["annotations"].append(
                {
                    "segmentation": [],
                    "iscrowd": 0,  # 0 or 1
                    "area": None,  # float or double
                    "image_id": imgName2Id[imgName],  # int
                    "bbox": None,  # list[float], [x,y,w,h]
                    "category_id": None,  # yolo class start from 0, coco start from 1
                    "id": annId,
                })
            annId += 1

    return Annotations, imgId2Name


def loadYOLO(classNames: dict, imagesPath: str, labelsPath: str):
    Annotations = {"info": {}, "licenses": [], "images": [], "annotations": [], "categories": []}

    classNames_list = list(classNames.keys())

    # categories
    for i, className in enumerate(classNames_list):
        Annotations["categories"].append(
            {"id": i + 1,
             "name": className,
             "supercategory": ""}
        )

    imgNameList = [f for f in os.listdir(imagesPath) if os.path.splitext(f)[1].lower() in IMG_FORMATS]
    labelNameList = [f for f in os.listdir(labelsPath) if os.path.splitext(f)[1].lower() == '.txt']

    imgIdIdx = 0
    imgId2Name, imgName2Id, name2ImgName = {}, {}, {}
    for imgName in imgNameList:
        imgId2Name[imgIdIdx] = imgName
        imgName2Id[imgName] = imgIdIdx
        name = os.path.splitext(imgName)[0]
        name2ImgName[name] = imgName
        imgIdIdx += 1

    annId = 0
    for txtName in tqdm(labelNameList, desc='YOLO to COCO...'):
        imgName = name2ImgName[txtName[:-4]]
        H, W, _ = cv2.imread(os.path.join(imagesPath, imgName)).shape

        # 写入image信息
        Annotations["images"].append(
            {
                "file_name": imgName,
                "height": H,
                "width": W,
                "id": imgName2Id[imgName]
            })

        count = 0
        with open(os.path.join(labelsPath, txtName), "r") as f:
            for line in f.readlines():
                if line == '\n':
                    continue

                split_ans = line.split()
                if len(split_ans) == 5:
                    # detect
                    cateId, center_x, center_y, w, h = split_ans
                    cateId = int(cateId)
                    labelName = classNames_list[cateId + 1]

                    center_x, center_y, w, h = float(center_x), float(center_y), float(w), float(h)
                    coco_xywh = yolo2coco([center_x, center_y, w, h], W, H)

                    # 可添加一些过滤条件，此处未添加

                    Annotations["annotations"].append(
                        {
                            "segmentation": [],
                            "iscrowd": 0,  # 0 or 1
                            "area": coco_xywh[2] * coco_xywh[3],  # float or double
                            "image_id": imgName2Id[imgName],  # int
                            "bbox": coco_xywh,  # list[float], [x,y,w,h]
                            "category_id": cateId + 1,  # yolo class start from 0, coco start from 1
                            "id": annId  # int
                        }
                    )
                else:
                    # segment

                    cateId = int(split_ans[0])
                    segmentation = []
                    x_min, y_min, x_max, y_max = W, H, 0, 0
                    for i in range(1, len(split_ans)):
                        if i % 2 == 1:
                            num = float(split_ans[i]) * W
                            x_min, x_max = min(num, x_min), max(num, x_max)
                        else:
                            num = float(split_ans[i]) * H
                            y_min, y_max = min(num, y_min), max(num, y_max)
                        segmentation.append(num)

                    coco_xywh = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
                    Annotations["annotations"].append(
                        {
                            "segmentation": [segmentation],
                            "iscrowd": 0,  # 0 or 1
                            "area": coco_xywh[2] * coco_xywh[3],  # float or double
                            "image_id": imgName2Id[imgName],  # int
                            "bbox": coco_xywh,  # list[float], [x,y,w,h]
                            "category_id": cateId + 1,  # yolo class start from 0, coco start from 1
                            "id": annId  # int
                        }
                    )

                annId += 1
                count += 1

        # 空标签处理
        if count == 0:
            Annotations["annotations"].append(
                {
                    "segmentation": [],
                    "iscrowd": 0,  # 0 or 1
                    "area": None,  # float or double
                    "image_id": imgName2Id[imgName],  # int
                    "bbox": None,  # list[float], [x,y,w,h]
                    "category_id": None,  # yolo class start from 0, coco start from 1
                    "id": None,
                })

    return Annotations, imgId2Name


def loadCOCO(classNames: dict, imagesPath: str, cocoJsonFile: str):
    with open(cocoJsonFile) as file:
        jsonInfo = json.load(file)

    imgId2Name = {}
    for image in jsonInfo['images']:
        imgId2Name[image['id']] = image['file_name']

    for index, anno in enumerate(jsonInfo['annotations']):
        # 可在此根据一些条件对标注信息进行过滤，如高宽比、面积等
        pass

    return jsonInfo, imgId2Name


# # 标签文件预处理相关
# def imgMask(maskInfo: dict, image_path, save_path, Type='mosaic', mirror=True):
#
#     # 初始化文件树结构
#     new_img_path = os.path.join(save_path, "images")
#     if os.path.exists(new_img_path):
#         new_img_path = os.path.join(save_path, "images_new")
#     os.makedirs(new_img_path)
#
#     imgId2Name = [f for f in os.listdir(image_path) if os.path.splitext(f)[1].lower() in IMG_FORMATS]
#     info = list(maskInfo.keys())
#     color = (114, 114, 114)
#
#     # mosaic操作
#     def _mosaic(im, size=(8, 8)):
#         h, w = im.shape[:2]
#         # size自适应, 避免区域过大导致块效应
#         sizeW, sizeH = max(min(int(w / 5), 15), 1), max(min(int(h / 5), 15), 1)
#         size = (sizeW, sizeH)
#         re = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
#         im = cv2.resize(re, (w, h), interpolation=cv2.INTER_AREA)
#         return im
#
#     # 羽化操作
#     def _gaussian(im):
#         h, w = im.shape[:2]
#         # 自适应高斯核, 尺寸必须为奇数
#         sizeW, sizeH = int(w / 5), int(h / 5)
#         sizeW = sizeW + 1 if sizeW % 2 == 0 else sizeW
#         sizeH = sizeH + 1 if sizeH % 2 == 0 else sizeH
#         size = (sizeW, sizeH)
#         roi = cv2.GaussianBlur(im, size, sigmaX=10, sigmaY=50)
#         return roi
#
#     for index in range(len(imgId2Name)):
#         img_name = imgId2Name[index]
#         if index in info:
#             img = cv2.imread(os.path.join(image_path, img_name))
#             imgCopy = img.copy()
#             for bbox in maskInfo[index]:
#                 # 坐标转换为VOC, 方便调用
#                 xyxy = coco2voc(bbox)
#                 if int(xyxy[1]) == int(xyxy[3]) or int(xyxy[0]) == int(xyxy[2]):
#                     # 避免小圆点导致打码报错
#                     continue
#                 if Type == 'gray':
#                     # 114色块填充打码
#                     left_top, right_bottom = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
#                     cv2.rectangle(imgCopy, left_top, right_bottom, color, -1)
#                 else:
#                     # 指定区域
#                     img_crops = img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
#                     # 图像对角线镜像
#                     img_crops = cv2.flip(img_crops, -1) if mirror else img_crops
#                     if Type == 'mosaic':
#                         # mosaic打码
#                         imgCopy[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] = _mosaic(img_crops)
#                     elif Type == 'Gaussian':
#                         # 高斯(羽化)
#                         imgCopy[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] = _gaussian(img_crops)
#
#             cv2.imwrite(os.path.join(new_img_path, img_name), imgCopy)
#         else:
#             shutil.copy(os.path.join(image_path, img_name), os.path.join(new_img_path, img_name))
            

