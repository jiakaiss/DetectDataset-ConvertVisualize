<div align='center' ><font size='10'><big><b> 数据集转换与可视化脚本</b></big></font></div>

# 功能

支持包括YOLO VOC COCO数据集互转，加载标注可视化功能。

# 框架
```
根目录
├── main.py         主程序 数据集互转和标注可视化  
├── config.py       记录类别信息
├── type2type.py    数据集文件格式转换
├── utils.py        坐标转换及标签文件格式初始化
├── visualize.py    标注可视化配置
└── load_data.py    读取标签文件
```



# 脚本调用

main.py负责数据集转换和标注可视化，选好所需功能后（格式转换或标签可视化），填好所需三个路径参数（分别为图像路径、标签路径和存储路径），运行python
 main.py即可
```
    parser = argparse.ArgumentParser(description="process some args")
    parser.add_argument('--mode', type=str, default='LabelView_COCO',
                        choices=['VOC2COCO', 'VOC2YOLO', 'YOLO2COCO', 'YOLO2VOC', 'COCO2YOLO', 'COCO2VOC',
                                 'LabelView_COCO', 'LabelView_VOC', 'LabelView_YOLO'],
                        help="the mode of changing label type to other type")
    
    parser.add_argument('--img_path', '-i', type=str,
                        default='',
                        help="the images Path of your dataset, eg: /home/yourPath/Images")
    
    parser.add_argument('--label_path', '-l', type=str,
                        default='',
                        help="label dirs, or it should be json file name if the type is COCO")
    
    parser.add_argument('--save_path', '-s', type=str,
                        default='',
                        help="the path to save the final labels or label-view")
```

