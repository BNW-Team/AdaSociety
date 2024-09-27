from PIL import Image
import os
import numpy as np

path=r"C:\Users\Hao Liu\Desktop\Study\2023 Summer\Research\tileset\tileset3\Alltile"
dirs = os.listdir(path)
cnt=0
for dir_ in dirs:
    filename = dir_ #原图地址及名称
    img = Image.open(path+"\\"+filename)
    size = img.size

    a=size[0]//48
    b=size[1]//48

    # 准备将图片切割成32*32张小图片
    weight = int(size[0] // a)
    height = int(size[1] // b)
    
    for j in range(a):
        for i in range(b):
            box = (weight * i, height * j, weight * (i + 1), height * (j + 1))
            region = img.crop(box)
            if np.array(region).mean()!=0 and np.array(region).mean()!=255:
            
                region.save('C:\\Users\\Hao Liu\\Desktop\\Study\\2023 Summer\\Research\\Projects\\bnw\\Pygame_test1\\resources\\cutted_tileset\\'+str(cnt+1)+".png")
                cnt+=1
    '''
    ————————————————
    版权声明：本文为CSDN博主「huavhuahua」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    原文链接：https://blog.csdn.net/huavhuahua/article/details/107030518
    '''

# 创建一个新的 RGBA 图像
image = Image.new('RGBA', (48, 48), (0, 0, 0, 0))

# 保存图片到本地文件
image.save('C:\\Users\\Hao Liu\\Desktop\\Study\\2023 Summer\\Research\\Projects\\bnw\\Pygame_test1\\resources\\cutted_tileset\\0.png')