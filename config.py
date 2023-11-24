# -------------------可视化------------------------#
#  功能: 类别过滤, 按类别输出bbox, 类别统计
#  参数: ① visClass: 需要可视化的类别, 不区分大小写
#                    yolo格式注意类别顺序, None视为过滤
#       ② bboxCrops: 是否单独保存bbox
# -----------------------------------------------#

# write your vis class
visClass = [
    'wangou', 'na', 'ti', 'pie',
    'piezhe', 'piedian', 'xiegouhuowogou', 'heng',
    'hengzhe', 'hengzhezhehuohengzhewan', 'hengzhezhezhe', 'hengzhezhezhegouhuohengpiewangou',
    'hengzhezhepie', 'hengzheti', 'hengzhegou', 'hengpiehuohenggou',
    'hengxiegou', 'dian', 'shu', 'shuwan',
    'shuwangou', 'shuzhezhegou', 'shuzhepiehuoshuzhezhe', 'shuti', 'shugou'
]
bboxCrops = True  # 方便审核误标情况

# include image suffixes
IMG_FORMATS = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pgm')


# write your class
Classes = {
    'wangou': [], 'na': [], 'ti': [], 'pie': [],
    'piezhe': [], 'piedian': [], 'xiegouhuowogou': [], 'heng': [],
    'hengzhe': [], 'hengzhezhehuohengzhewan': [], 'hengzhezhezhe': [], 'hengzhezhezhegouhuohengpiewangou': [],
    'hengzhezhepie': [], 'hengzheti': [], 'hengzhegou': [], 'hengpiehuohenggou': [],
    'hengxiegou': [], 'dian': [], 'shu': [], 'shuwan': [],
    'shuwangou': [], 'shuzhezhegou': [], 'shuzhepiehuoshuzhezhe': [], 'shuti': [],
    'shugou': [],
}



