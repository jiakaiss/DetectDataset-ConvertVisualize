import xml.etree.ElementTree as ET

"""
格式转换, 以下是各格式的特点
yolo 归一化中心点xywh
voc  左下角右上角xyxy
coco 左下角xywh
"""


def yolo2coco(yolo_xywh: list, imgW: int, imgH: int):
    # 中心点xywh 转换为 左下角xywh
    # 反归一化
    center_x = yolo_xywh[0] * imgW
    center_y = yolo_xywh[1] * imgH
    w = yolo_xywh[2] * imgW
    h = yolo_xywh[3] * imgH

    x_min = max(center_x - w / 2, 0)
    y_min = max(center_y - h / 2, 0)
    coco_xywh = [x_min, y_min, w, h]
    return coco_xywh


def coco2yolo(coco_xywh: list, imgW: int, imgH: int):
    # 左下角xywh 转换为 中心点xywh
    center_x = coco_xywh[0] + coco_xywh[2] / 2
    center_y = coco_xywh[1] + coco_xywh[3] / 2

    # 归一化
    x, y = center_x / imgW, center_y / imgH
    w, h = coco_xywh[2] / imgW, coco_xywh[3] / imgH
    yolo_xywh = '{} {} {} {}'.format(x, y, w, h)
    return yolo_xywh


def voc2coco(xyxy: list):
    # xyxy2xywh
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]

    x_min = xyxy[0]
    y_min = xyxy[1]
    coco_xywh = [x_min, y_min, w, h]
    return coco_xywh


def coco2voc(coco_xywh: list):
    # xywh2xyxy
    w = coco_xywh[2]
    h = coco_xywh[3]

    x_min = coco_xywh[0]
    y_min = coco_xywh[1]
    x_max = x_min + w
    y_max = y_min + h
    xyxy = [x_min, y_min, x_max, y_max]
    return xyxy


#   ------------------------------------------------------------------------------------    #
# 生成xml
def generateImgInfo(imagesPath, imgName, imgH, imgW, imgDepth):
    # generate the xml content
    root = ET.Element('annotation')
    root.text = '\n\t'
    # folder
    folder = ET.SubElement(root, 'folder')
    folder.text = imagesPath
    folder.tail = '\n\t'
    # filename
    filename = ET.SubElement(root, 'filename')
    filename.text = imgName
    filename.tail = '\n\t'
    # source
    source = ET.SubElement(root, 'source')
    source.text = '\n\t'
    database = ET.SubElement(source, 'database')
    database.text = 'not important'
    database.tail = '\n\t'
    annotation = ET.SubElement(source, 'annotation')
    annotation.text = 'not important'
    annotation.tail = '\n\t'
    image = ET.SubElement(source, 'image')
    image.text = 'not important'
    image.tail = '\n\t'
    flickrid = ET.SubElement(source, 'flickrid')
    flickrid.text = 'not important'
    flickrid.tail = '\n\t'

    # img size
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(imgW)
    height = ET.SubElement(size, 'height')
    height.text = str(imgH)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(imgDepth)

    segmented = ET.SubElement(root, 'segmented')
    segmented.text = '0'  # only for object detection

    indentFormat(root)

    tree = ET.ElementTree(root)
    return tree


def insertObject(tree, cateName, bbox):
    """
    <object>
        <name>className</name>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>0</xmin>
            <ymin>0</ymin>
            <xmax>0</xmax>
            <ymax>0</ymax>
        </bndbox>
    </object>
    """
    root = tree.getroot()
    object = ET.SubElement(root, 'object')

    name = ET.SubElement(object, 'name')
    name.text = cateName
    truncated = ET.SubElement(object, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(object, 'difficult')
    difficult.text = '0'

    bndbox = ET.SubElement(object, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = str(bbox[0])
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = str(bbox[1])
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = str(bbox[0] + bbox[2])
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = str(bbox[1] + bbox[3])

    indentFormat(root)
    tree = ET.ElementTree(root)
    return tree


def indentFormat(elem, level=0):
    # 格式美化
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indentFormat(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# def splitData(datasetPath, train_percent, valtest_percent, backups=False, backup_path=None):
#     import os
#     import random
#     imgPath = os.path.join(datasetPath, 'images')
#
#     imgs = os.listdir(imgPath)
#     numLabel = len(imgs)
#     tv = int(numLabel * (1.0 - train_percent))
#     tt = int(numLabel * (1.0 - train_percent) * valtest_percent)
#
#     # 随机划分
#     valtest = random.sample(range(numLabel), tv)  # 从所有list划分val和test
#     test = random.sample(valtest, tt)  # 从valtest划分test
#
#     if backups:
#         datasetPath = backup_path
#         os.makedirs(datasetPath, exist_ok=True)
#     train_txt = os.path.join(datasetPath, 'train.txt')
#     val_txt = os.path.join(datasetPath, 'val.txt')
#     test_txt = os.path.join(datasetPath, 'test.txt')
#     ftrain = open(train_txt, 'w')
#     fval = open(val_txt, 'w')
#     ftest = open(test_txt, 'w')
#     for i in range(numLabel):
#         name = './images/' + imgs[i] + '\n'
#         if i in valtest:
#             if i in test:
#                 ftest.write(name)
#             else:
#                 fval.write(name)
#         else:
#             ftrain.write(name)
#     ftrain.close()
#     fval.close()
#     ftest.close()
#     return train_txt, val_txt, test_txt