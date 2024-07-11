from PIL import Image
import os

# 设置新的图片尺寸
new_width = 640
new_height = 192

# 设置图片所在的目录
# directory = '/home/lsk/Dataset/kitti_raw/2011_09_30/2011_09_30_drive_0020_sync/image_02/data/'
directory = '/home/lsk/MonocularDepthEsitimation/MonoGS/datasets/kitti/3020_m/input'
output = '/home/lsk/MonocularDepthEsitimation/MonoGS/datasets/kitti/3020_m/1'

a = range(977, 1041)
b = range(150, 214)

# 遍历目录中的所有文件 并调整大小
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        # 打开图片
        img = Image.open(os.path.join(directory, filename))
        # 去前四位
        filename = filename[5:]
        # 调整图片大小
        # img = img.resize((new_width, new_height), Image.ANTIALIAS)
        # 保存新的图片
        img.save(os.path.join(output, "0000{}".format(filename)))
        print("Resized image:", filename)


