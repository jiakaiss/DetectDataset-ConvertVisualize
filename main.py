import os
import argparse
import config
import type2type
import load_data


def parse_opt():
    parser = argparse.ArgumentParser(description="process some args")
    parser.add_argument('--mode', type=str, default='',
                        choices=['VOC2COCO', 'VOC2YOLO', 'YOLO2COCO', 'YOLO2VOC', 'COCO2YOLO', 'COCO2VOC',
                                 'LabelView_COCO', 'LabelView_VOC', 'LabelView_YOLO'],
                        help="the mode of changing label type to other type")

    parser.add_argument('--img_path', '-i', type=str,
                        default='/Users/wang/Desktop/datasets/handwritten/images',
                        help="the images Path of your dataset, eg: /home/yourPath/Images")
    
    parser.add_argument('--label_path', '-l', type=str,
                        default='/Users/wang/Desktop/save_/labels',
                        help="label dirs, or it should be json file name if the type is COCO")
    
    parser.add_argument('--save_path', '-s', type=str,
                        default='/Users/wang/Desktop/new',
                        help="the path to save the final labels or label-view")
    
    # parser.add_argument('--split', type=tuple, default=None,  # (0.95, 1) None
    #                     help="random split dataset based on the ratio, (train, test in valtest)")

    return parser.parse_args()


def main(opt):
    os.makedirs(opt.save_path, exist_ok=True)
    if opt.mode in ['LabelView_COCO', 'LabelView_VOC', 'LabelView_YOLO']:
        from visualize import label_view
        label_view(opt.img_path, opt.label_path, opt.save_path, opt.mode, config.visClass, config.bboxCrops)
        return

    # load data
    loadFunc = getattr(load_data, 'load' + opt.mode.split('2')[0])
    jsonInfo, imgId2Name = loadFunc(config.Classes, opt.img_path, opt.label_path)

    # COCO to ...
    convertFunc = getattr(type2type, 'COCO2' + opt.mode.split('2')[1])
    convertFunc(jsonInfo, opt.save_path, opt.img_path, imgId2Name)

    # if opt.split is not None and '2COCO' not in opt.mode:
    #     splitData(opt.save_path, 0.9, 0)


if __name__ == '__main__':
    opts = parse_opt()
    main(opts)
