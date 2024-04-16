import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pyttsx4

# pyttsx4 for the text to speech
engine = pyttsx4.init()
engine.say('Welcome Sir Im the deep learning model. Please Wait for the result till i evaluate the process')
engine.runAndWait()

# Load the data 
data = tf.keras.utils.image_dataset_from_directory(os.path.join('modelFiles', 'data'))
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
batch[0].shape
   

# Preprocessing the data
dataPre = data.map(lambda x, y: (x/255, y))
see = dataPre.as_numpy_iterator().next()[0].max()
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])


# Split the data
trainSize = int(len(dataPre)* .5)
valSize = int(len(dataPre)* .3)+1
testSize = int(len(dataPre)*.2)+1
train = dataPre.take(trainSize)
val = dataPre.skip(trainSize).take(valSize)
test = dataPre.skip(trainSize+valSize).take(testSize)


# Build Deep Learning Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256, 3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, (3,3), 1, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(16, (3,3), 1, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    tf.keras.layers.Dropout(0.2)
])

# Comile the model
model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])


# train the model
'''
logdir = 'logs'   # Exception if u want to generate the calllogs file for later checks can uncomment this
tensorboardcall = tf.keras.callbacks.TensorBoard(log_dir=logdir)
'''
hist = model.fit(train, epochs=25, validation_data=val)

# plot the performance(Exception case)
'''
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper right')
plt.show()
'''


# Evaluation of the model
pre = tf.keras.metrics.Precision()
re = tf.keras.metrics.Recall()
acc = tf.keras.metrics.BinaryAccuracy()

for batch in test.as_numpy_iterator():
    x, y = batch
    yhat = model.predict(x)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)


# test the model now
img = cv2.imread('C:\\Users\\MILAN\\Downloads\\gojo.png')
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))

ye = model.predict(np.expand_dims(resize/255, 0))

if ye > 0.5:
    print('sad person')
    engine.say('According to my analysis Person is sad')
else:
    print('happy person')
    engine.say('According to my analysis Person is happy')

engine.runAndWait()



