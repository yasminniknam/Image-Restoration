import re 
import os
import cv2 as cv
import numpy as np

def getPSNR(I1, I2):
    s1 = cv.absdiff(I1, I2) #|I1 - I2|
    s1 = np.float32(s1)     # cannot make a square on 8 bits
    s1 = s1 * s1            # |I1 - I2|^2
    sse = s1.sum()          # sum elements per channel
    if sse <= 1e-10:        # sum channels
        return 0            # for small values return zero
    else:
        shape = I1.shape
        mse = 1.0 * sse / (shape[0] * shape[1] * shape[2])
        psnr = 10.0 * np.log10((255 * 255) / mse)
        return psnr


address = "/content/photo_restoration/test_videos/recovered_frames/final_output/"
imagelist = os.listdir(address)
imagelist.sort(key=lambda f: int(re.sub('\D', '', f)))
# print(imagelist)
image1 = cv.imread(address+'1.jpg')
image2 = cv.imread(address+'2.jpg')
print(getPSNR(image1, image2))