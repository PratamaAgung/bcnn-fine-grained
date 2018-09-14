import cv2
import glob
import numpy as np

images_mean = []
for image_path in glob.glob("data_gemastik/*/*"):
    print(image_path)
    image_data = cv2.imread(image_path)
    # print(image_data.shape)
    image_mean = np.mean(image_data, axis=(0,1))
    # print(image_mean)
    images_mean.append(image_mean)

result_mean = np.mean(images_mean, axis= 0)
print(result_mean)
