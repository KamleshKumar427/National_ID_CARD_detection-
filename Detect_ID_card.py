import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
import cv2
from PIL import Image
import sys

np.set_printoptions(threshold=sys.maxsize)

sample_image=cv2.imread('/content/s3.jpg')


# OTSU segmentation
img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)

img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
thresh = threshold_otsu(img_gray)
img_otsu  = img_gray < thresh

plt.axis('off')
# plt.imshow(img_otsu)

def filter_image(image, mask):

    r = image[:,:,0] * mask
    g = image[:,:,1] * mask
    b = image[:,:,2] * mask

    return np.dstack([r,g,b])

filtered = filter_image(img, img_otsu)

plt.axis('off')
# plt.imshow(filtered)

img2Arr=np.asarray(Image.fromarray(filtered).convert('L'))
plt.imshow(img2Arr,)


#color range segmentation
img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)


low = np.array([102, 194, 164])
high = np.array([229,245,249])
# low = np.array([125,125,125])
# high = np.array([160,160,160])

mask = cv2.inRange(img, low, high)

plt.axis('off')
plt.imshow(mask)

img3Arr = mask

# edge detection segmentation
img=cv2.cvtColor(sample_image,cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)
image = cv2.erode(img, kernel)
img_blur = cv2.GaussianBlur(img, (3,3), 0)
for i in range(10):
    img_blur = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
plt.axis('off')
img1Arr = img_blur // 255

plt.axis('off')
plt.imshow(img1Arr)



# creating new image from three segmentation techniques
finalImgArr = np.ones(img1Arr.shape)*255


for h in range(img1Arr.shape[0]):
  for w in range(img1Arr.shape[1]):
    if(img1Arr[h][w] != 0 and img2Arr[h][w] == 0 and img3Arr[h][w] == 255 ):
      finalImgArr[h][w] = 0

plt.axis('off')
plt.imshow(finalImgArr,cmap='gray')


# author: Kamlesh kumar
# linkedIn: https://www.linkedin.com/in/kamlesh-kumar-389847224/
