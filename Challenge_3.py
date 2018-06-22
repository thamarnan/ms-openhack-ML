# coding: utf-8
# In[180]:
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image
# In[181]:

file_path = "/home/team3user/new_images/hardshell_jackets/10116634x1072858_zm.jpeg";
new_file_path = "/home/team3user/10116634x1072858_zm.png";
new_file_nor = "/home/team3user/10116634x1072858_zm_nor.png"
#img=mpimg.imread('/home/team3user/gear_images/helmets/897876.jpeg')

def histeq(im,nbr_bins=256):
  """  Histogram equalization of a grayscale image. """

  # get image histogram
  imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
  cdf = imhist.cumsum() # cumulative distribution function
  cdf = 255 * cdf / cdf[-1] # normalize

  # use linear interpolation of cdf to find new pixel values
  im2 = interp(im.flatten(),bins[:-1],cdf)

  return im2.reshape(im.shape), cdf

def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255/(maxval-minval)) 
            arr[...,i] += 128
    return arr

def do_normalize(file_path,new_file_path):
    img = Image.open(file_path).convert('RGB')
    arr = np.array(img)
    np.save(new_file_path,normalize(arr).astype('uint8'))
    #new_img = Image.fromarray(normalize(arr).astype('uint8'),'RGB')
    #new_img.save(new_file_path)




# In[182]:


from PIL import Image
from numpy import *
im1 = Image.open(file_path).convert('L')
im = array(im1)
im2,cdf = histeq(im)
im1.show()
img = Image.fromarray(im2, 'RGBA')
img.save(new_file_path)
pil_im1 = Image.open(file_path, 'r')
imshow(np.asarray(pil_im1))


# In[175]:


pil_im2 = Image.open(new_file_path, 'r')
imshow(np.asarray(pil_im2))


# In[176]:


#Original file
pil_im = Image.open(file_path, 'r')
imshow(np.asarray(pil_im))


# In[186]:


#Histgram for the original file
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(file_path,0)
plt.hist(img.ravel(),bins=255); plt.show()


# In[183]:


#Normalize the reshaped file
do_normalize(file_path, new_file_nor)
pil_im = Image.open(new_file_nor, 'r')
imshow(np.asarray(pil_im))


# In[187]:


#Histogram the reshaped file
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(new_file_nor,0)
plt.hist(img.ravel(),256,[0,256]); plt.show()


# In[188]:


#Normalize all images. 
import os
path = "/home/team3user/new_images/" 
def list_files(directory,prefix = ''):
    for filename in os.listdir(directory):
        if  filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"): 
            old_file_path = os.path.join(directory, filename)
            #print(old_file_path)
            new_file_path = old_file_path.replace("new_images","normalized_images")
            new_path = os.path.dirname(new_file_path)
            #print(new_path)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            print(new_file_path)
            do_normalize(old_file_path, new_file_path)
            continue            
        else:
            continue
dir_list = os.listdir(path)
for p in dir_list:
    sub_path = os.path.join(path , p)
    print(sub_path)
    list_files(sub_path)

