import cv2
import numpy as np
import random as rnd
#import seaborn as sns 
from keras.callbacks import ModelCheckpoint
from make_model import *
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img, img_to_array, load_img#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

seed = 11
rnd.seed(seed)
np.random.seed(seed)

model = make_model()
epochs = 30
winH,winW = 50,50

############################

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
		rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(winH, winW),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(winH, winW),
        batch_size=batch_size,
        class_mode='binary')

filepath="weights_best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

class_weight = {0: 10,
                1: 1}
				
#model.fit_generator(
       # train_generator,
       # steps_per_epoch=5131 // batch_size,
        #epochs=epochs,
        #validation_data=validation_generator,
        #validation_steps=1603 // batch_size,
        #callbacks=callbacks_list,
        #class_weight=class_weight)
history = model.fit_generator(
        train_generator,
        steps_per_epoch=5131 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=1603 // batch_size,
        callbacks=callbacks_list,
        class_weight=class_weight)

print(type(history))
loss = history.history['val_loss']
loss
from matplotlib import pyplot as plt
#%matplotlib inline
epoch_count = range(1, len(loss) + 1)

# Visualize loss history
plt.plot(epoch_count, loss, 'r')
plt.xlabel('Epoch')
plt.ylabel('Loss')

acc = history.history['val_accuracy']
acc
from matplotlib import pyplot as plt
#%matplotlib inline
epoch_count = range(1, len(acc) + 1)

# Visualize loss history
plt.plot(epoch_count, acc, 'b')
plt.xlabel('Epoch')
plt.ylabel('Accuracy in blue and loss in red')
plt.show();

