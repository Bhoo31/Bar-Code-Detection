#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import os.path as op
import numpy as np
import cv2
import matplotlib.pylab as plt
import matplotlib.pylab as pylab


# In[2]:


image1 = cv2.imread(r"C:\Users\bhoom\Downloads\bar1.jpeg")
image2 = cv2.imread(r"C:\Users\bhoom\Downloads\bar2.jpeg")
image3 = cv2.imread(r"C:\Users\bhoom\Downloads\bar3.jpeg")
image4 = cv2.imread(r"C:\Users\bhoom\Downloads\bar4.jpeg")


# In[3]:


def plotimage(image, title):
    plt.imshow(image,cmap=plt.get_cmap('gray')), plt.title(title, size=20), plt.axis('off')

# read in and display image. Note that there are 4 bar code images provided
#img = cv2.imread('bar_code1.jpg')
image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB) # convert bgr to rgb
image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2RGB) # convert bgr to rgb
image3 = cv2.cvtColor(image3,cv2.COLOR_BGR2RGB) # convert bgr to rgb
image4 = cv2.cvtColor(image4,cv2.COLOR_BGR2RGB) # convert bgr to rgb


# In[4]:


plt.figure(figsize=(15,15))
plt.subplot(221), plotimage(image1, 'Image_1')
plt.subplot(222), plotimage(image2, 'Image_2')
plt.subplot(223), plotimage(image3, 'Image_3')
plt.subplot(224), plotimage(image4, 'Image_4')
plt.tight_layout()
plt.show()


# In[5]:


# convert to greyscale using opencv cvtColor
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray_image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
gray_image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
#gray_img=gray_img.astype(np.int32)
#cv2.imshow('Grayscale', gray_img)
# display the greyscale image
plt.figure(figsize=(15,15))
plt.subplot(221), plotimage(gray_image1, 'Image_1')
plt.subplot(222), plotimage(gray_image2, 'Image_2')
plt.subplot(223), plotimage(gray_image3, 'Image_3')
plt.subplot(224), plotimage(gray_image4, 'Image_4')
plt.tight_layout()
plt.show()


# In[6]:


#calculate x & y gradient using Sobel edge detector

sobelx = cv2.Sobel(gray_image1, cv2.CV_8UC1, 1, 0, ksize=5)
sobely = cv2.Sobel(gray_image1, cv2.CV_8UC1, 0, 1, ksize=5)

sobelx2 = cv2.Sobel(gray_image2, cv2.CV_8UC1, 1, 0, ksize=5)
sobely2 = cv2.Sobel(gray_image2, cv2.CV_8UC1, 0, 1, ksize=5)

sobelx3 = cv2.Sobel(gray_image3, cv2.CV_8UC1, 1, 0, ksize=5)
sobely3 = cv2.Sobel(gray_image3, cv2.CV_8UC1, 0, 1, ksize=5)

sobelx4 = cv2.Sobel(gray_image4, cv2.CV_8UC1, 1, 0, ksize=5)
sobely4 = cv2.Sobel(gray_image4, cv2.CV_8UC1, 0, 1, ksize=5)
# subtract the y-gradient from the x-gradient
edges_1 = cv2.subtract(sobelx, sobely)

edges_2 = cv2.subtract(sobelx2, sobely2)

edges_3 = cv2.subtract(sobelx3, sobely3)

edges_4 = cv2.subtract(sobelx4, sobely4)

# scale the image using convertScaleAbs function

edges_1 = cv2.convertScaleAbs(edges_1)

edges_2 = cv2.convertScaleAbs(edges_2)

edges_3 = cv2.convertScaleAbs(edges_3)

edges_4 = cv2.convertScaleAbs(edges_4)

# display the gradient image
plt.figure(figsize=(15,15))
plt.subplot(221), plotimage(edges_1, 'Image_1')
plt.subplot(222), plotimage(edges_2, 'Image_2')
plt.subplot(223), plotimage(edges_3, 'Image_3')
plt.subplot(224), plotimage(edges_4, 'Image_4')
plt.tight_layout()
plt.show()


# In[7]:


# blur the image
blurred_1 = cv2.blur(edges_1, (5, 5))
blurred_2 = cv2.blur(edges_2, (5, 5))
blurred_3 = cv2.blur(edges_3, (5, 5))
blurred_4 = cv2.blur(edges_4, (5, 5))



# display the gradient image
plt.figure(figsize=(15,15))
plt.subplot(221), plotimage(blurred_1, 'Image_1')
plt.subplot(222), plotimage(blurred_2, 'Image_2')
plt.subplot(223), plotimage(blurred_3, 'Image_3')
plt.subplot(224), plotimage(blurred_4, 'Image_4')
plt.tight_layout()
plt.show()


# In[8]:


# threshold the image
#thresh_value = 128
#max_value = 225
_, binary_1 = cv2.threshold(blurred_1,128 ,225, cv2.THRESH_BINARY)
_, binary_2 = cv2.threshold(blurred_2,150,255, cv2.THRESH_BINARY)
_, binary_3 = cv2.threshold(blurred_3,101,255, cv2.THRESH_BINARY)
_, binary_4 = cv2.threshold(blurred_4,128,255, cv2.THRESH_BINARY)

# display the thresholded image
plt.figure(figsize=(15,15))
plt.subplot(221), plotimage(binary_1, 'Image_1')
plt.subplot(222), plotimage(binary_2, 'Image_2')
plt.subplot(223), plotimage(binary_3, 'Image_3')
plt.subplot(224), plotimage(binary_4, 'Image_4')
plt.tight_layout()
plt.show()


# In[9]:


# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
closed_1 = cv2.morphologyEx(binary_1, cv2.MORPH_CLOSE, kernel)
closed_2 = cv2.morphologyEx(binary_2, cv2.MORPH_CLOSE, kernel)
closed_3 = cv2.morphologyEx(binary_3, cv2.MORPH_CLOSE, kernel)
closed_4 = cv2.morphologyEx(binary_4, cv2.MORPH_CLOSE, kernel)

# perform a series of erosions and dilations
eroded_1 = cv2.erode(closed_1 , kernel, iterations=5)
dilated_1 = cv2.dilate(eroded_1, kernel, iterations=5)


eroded_2 = cv2.erode(closed_2, kernel, iterations=3)
dilated_2 = cv2.dilate(eroded_2, kernel, iterations=3)


eroded_3 = cv2.erode(closed_3, kernel, iterations=5)
dilated_3 = cv2.dilate(eroded_3, kernel, iterations=5)


eroded_4 = cv2.erode(closed_4, kernel, iterations=5)
dilated_4 = cv2.dilate(eroded_4, kernel, iterations=5)


# display the thresholded image
plt.figure(figsize=(15,15))
plt.subplot(221), plotimage(dilated_1, 'Image_1')
plt.subplot(222), plotimage(dilated_2, 'Image_2')
plt.subplot(223), plotimage(dilated_3, 'Image_3')
plt.subplot(224), plotimage(dilated_4, 'Image_4')
plt.tight_layout()
plt.show()


# In[10]:


# Find contours
contours_1, hierarchy_1 = cv2.findContours(dilated_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_2, hierarchy_2 = cv2.findContours(dilated_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_3, hierarchy_3 = cv2.findContours(dilated_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_4, hierarchy_4 = cv2.findContours(dilated_4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours by size using cv2.contourArea
contours_1 = sorted(contours_1, key=cv2.contourArea, reverse=True)
contours_2 = sorted(contours_2, key=cv2.contourArea, reverse=True)
contours_3 = sorted(contours_3, key=cv2.contourArea, reverse=True)
contours_4 = sorted(contours_4, key=cv2.contourArea, reverse=True)

# Compute the rotated bounding box of the largest contour, if any
def get_largest_bounding_box(contours):
    if contours:
        largest_contour = contours[0]
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box
    return None

box1 = get_largest_bounding_box(contours_1)
box2 = get_largest_bounding_box(contours_2)
box3 = get_largest_bounding_box(contours_3)
box4 = get_largest_bounding_box(contours_4)

# Draw a bounding box around the detected barcode and display the image
if box1 is not None:
    result_1 = cv2.drawContours(image1, [box1], 0, (0, 0, 255), 1)
if box2 is not None:
    result_2 = cv2.drawContours(image2, [box2], 0, (0, 0, 255), 2)
if box3 is not None:
    result_3 = cv2.drawContours(image3, [box3], 0, (0, 0, 255), 3)
if box4 is not None:
    result_4 = cv2.drawContours(image4, [box4], 0, (0, 0, 255), 4)

# Display the thresholded image
plt.figure(figsize=(15, 15))
if box1 is not None:
    plt.subplot(221), plotimage(result_1, 'Bar_Code1 Image')
if box2 is not None:
    plt.subplot(222), plotimage(result_2, 'Bar_Code2 image')
if box3 is not None:
    plt.subplot(223), plotimage(result_3, 'Bar_Code3 image')
if box4 is not None:
    plt.subplot(224), plotimage(result_4, 'Bar_Code4 image')
plt.tight_layout()
plt.show()



# In[ ]:




