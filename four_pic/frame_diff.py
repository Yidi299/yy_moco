import cv2
import numpy as np

# 加载四张连续的图片
image_filenames = ["image1.png", "image2.png", "image3.png", "image4.png"]
images = [cv2.imread(filename) for filename in image_filenames]

# 计算帧差
frame_diffs = [cv2.absdiff(images[i], images[i+1]) for i in range(len(images) - 1)]

# 将帧差添加为第四个通道
images_with_diff = []

for i in range(len(frame_diffs)):
    # 计算灰度帧差
    gray_diff = cv2.cvtColor(frame_diffs[i], cv2.COLOR_BGR2GRAY)
    
    # 将灰度帧差归一化到[0, 1]区间
    normalized_diff = gray_diff.astype(np.float32) / 255.0

    # 将归一化的帧差添加为第四个通道
    image_with_diff = cv2.merge((images[i], normalized_diff))
    images_with_diff.append(image_with_diff)

# 至此，images_with_diff列表中的每个元素都是一个带有四个通道的图像（B, G, R, 帧差）