
# coding: utf-8

# In[21]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from imutils import paths
import imutils
import cv2
import os
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras import backend as K
import numpy as np


# In[22]:


def image_to_feature_vector(image, size=(128, 128)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size)
#.flatten()

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images("/home/team3user/gear_images/"))

# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = [] 
labels = []
labelMap = {}
count = 0
def process_image(imagePath):
    global count
    # Process one image
    # update the raw images, features, and labels matricies,
    # respectively
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-2]
    if label not in labelMap.keys():
        labelMap[label] = count
        label = count
        count += 1
    else:
        label = labelMap[label] 
    #print(imagePath.split(os.path.sep)[-2],imagePath)
    # extract raw pixel intensity "features", followed by a color
    # histogram to characterize the color distribution of the pixels
    # in the image
    pixels = image_to_feature_vector(image) 
    rawImages.append(pixels) 
    labels.append([label])
    

# loop over the input images

for (i, imagePath) in enumerate(imagePaths):
    process_image(imagePath)
    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))



# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages) 
labels = np.array(labels)
num_classes = len(labelMap)
print("[INFO] label mapping:",labelMap,num_classes)
print("[INFO] pixels matrix: {:.2f}MB".format(
    rawImages.nbytes / (1024 * 1000.0))) 

(trainRI, testRI, trainRL, testRL) = train_test_split(
    rawImages, labels, test_size=0.25, random_state=42)

print("[INFO] trainRL Dataset shape",min(trainRL))
x_train = trainRI
y_train = keras.utils.to_categorical( trainRL , num_classes)
x_test = testRI
y_test = keras.utils.to_categorical(testRL, num_classes)
print("[INFO] x_train Dataset shape",x_train.shape)
print("[INFO] y_train categorical result shape",y_train.shape)
print("[INFO] x_test Dataset shape",x_test.shape)
print("[INFO] y_test categorical result shape",y_test.shape)


# In[23]:


input_shape = x_train.shape[1:]
print("[INFO] input_shape:",str(input_shape))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
filter_size = (4,4)
model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(64, filter_size, activation='relu', input_shape=(128, 128, 3)))
model.add(Conv2D(64, filter_size, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, filter_size, activation='relu'))
model.add(Conv2D(128, filter_size, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model.summary())


# In[ ]:


print("[INFO] x_train Dataset shape",x_train.shape)
print("[INFO] y_train categorical result shape",y_train.shape)
print(y_train) 
model.fit(x_train, y_train, batch_size=128, epochs=20)
model.save_weights('/home/team3user/my_model_weights.h5')
model.save('/home/team3user/my_model.h5')  # creates a HDF5 file 'my_model.h5'


# In[20]:


score = model.evaluate(x_test, y_test, verbose=0, batch_size=128)
for name,value in zip(model.metrics_names, score):
    print(name, value) 

#print(model.metrics_names)


# In[19]:


image_path = "/home/team3user/test_images/tents/880786.jpeg"
image_path = "/home/team3user/test_images/helmets/2039689_101_main.jpg"
image = image_to_feature_vector(image = cv2.imread(image_path))
result = model.predict(np.array([image]))
print(labelMap)
#print(y_train)
print(result)
print("I'm sure it's a ",list(labelMap.keys())[list(labelMap.values()).index(np.where(result[0]==1)[0][0])]) # Prints george

