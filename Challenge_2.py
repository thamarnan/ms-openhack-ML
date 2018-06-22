%matplotlib inline 
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import skimage.color as sc
import glob
import os
size = 128, 128

rootdir = "/home/team3user/gear_images/*/*"
for filename in glob.iglob(rootdir, recursive=True):
    im = Image.open(filename)
    im.thumbnail(size)
    layer = Image.new('RGB', (128, 128), (0, 0, 0))
    myW, myH = im.size
    layer.paste(im, ((128-myW)//2,(128-myH)//2))
    newpath=filename.replace("gear_images","new_images")
    new_dir = os.path.dirname(newpath)
    print(new_dir)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    layer.save(newpath)
