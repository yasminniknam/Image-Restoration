import re 
import os
import sys
import cv2 as cv
import numpy as np
import skimage.io as io
from skimage import img_as_ubyte
from PIL import Image, ImageFile

sys.path.insert(0, '/content/photo_restoration/Global')
import detection
sys.path.remove('/content/photo_restoration/Global')

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

def irregular_hole_synthesize(img, new_img, mask, new_img2=None):

    img_np = np.array(img).astype("uint8")
    new_img_np = np.array(new_img).astype("uint8")
    mask_np = np.array(mask).astype("uint8")
    mask_np = mask_np / 255
    if new_img2 != None:
      new_img2_np = np.array(new_img).astype("uint8")
      img_new = img_np * (1 - mask_np) + mask_np * (new_img_np+new_img2_np)/2.
    else:
      img_new = img_np * (1 - mask_np) + mask_np * new_img_np

    hole_img = Image.fromarray(img_new.astype("uint8")).convert("RGB")

    return hole_img

address = "/content/photo_restoration/test_videos/recovered_frames/final_output/"
final_address = "/content/photo_restoration/test_videos/recovered_frames/after_new_stage/"
imagelist = os.listdir(address)
imagelist.sort(key=lambda f: int(re.sub('\D', '', f)))
# print(imagelist)
# image1 = cv.imread(address+'140.jpg')
# image2 = cv.imread(address+'101.jpg')
# image3 = cv.imread(address+'105.jpg')
# print(getPSNR(image1.copy(), image2.copy()))
# print(getPSNR(image2.copy(), image3.copy()))

image = [cv.imread(address+imagelist[i]) for i in range(len(imagelist))]
image_mask = [Image.open(address+imagelist[i]).convert("RGB") for i in range(len(imagelist))]

stage_1_input_dir = ""
mask_dir = ""
gpu1 = 0
input_opts_stage1 = ["--test_path", stage_1_input_dir, "--output_dir", mask_dir,
                                    "--input_size", "scale_256", "--GPU", gpu1]

os.chdir("./Global")
_ , mask = detection.detection(input_opts_stage1, image_mask, imagelist)
os.chdir("../")

if not os.path.exists(final_address):
      os.makedirs(final_address)

for i in range(len(image)):
  prev = -1
  next = 0
  chosen = -1
#   print("* frame "+str(i)+" *")
  if i == 0:
    next = getPSNR(image[i], image[i+1])
    chosen = (next, i+1)
    # print(getPSNR(image[i], image[i+1]))
    # print("**")
    # continue
  elif i == len(image)-1:
    prev = getPSNR(image[i], image[i-1])
    chosen = (prev, i-1)
    # print(getPSNR(image[i], image[i-1]))
    # print("**")
    # continue
  else:
    prev = getPSNR(image[i-1], image[i])
    next = getPSNR(image[i], image[i+1])
    # if next > prev:
    #   chosen = (next, i+1)
    # else:
    #   chosen = (prev,i-1)
    

    if next > 19:
      if prev > 19:
        image_mask[i] = irregular_hole_synthesize(image_mask[i], image_mask[i+1], mask[i], new_img2=i-1)
      else:
        image_mask[i] = irregular_hole_synthesize(image_mask[i], image_mask[i+1], mask[i])
    elif prev > 19:
      image_mask[i] = irregular_hole_synthesize(image_mask[i], image_mask[i-1], mask[i])

    # print(getPSNR(image[i-1], image[i]))
    # print(getPSNR(image[i], image[i+1]))
    # print("**")
  # if chosen[0] > 19:
  #   image_mask[i] = irregular_hole_synthesize(image_mask[i], image_mask[chosen[1]], mask[i])
    
  io.imsave(os.path.join(final_address, imagelist[i]), img_as_ubyte(image_mask[i]))
      
