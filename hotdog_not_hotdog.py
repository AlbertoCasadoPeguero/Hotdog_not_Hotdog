#Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#Preprocessing training set for data augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 180,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip = True)

training_set = train_datagen.flow_from_directory(
        'hotdog/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#Preprocessing the test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
        'hotdog/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#Building the CNN
cnn = tf.keras.models.Sequential()

#First layer
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64,64,3]))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size = 3,strides = 2))

#Second layers
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size = 3,strides = 2))

#Flattening
cnn.add(tf.keras.layers.Flatten())

#Fully connection
cnn.add(tf.keras.layers.Dense(units = 256, activation = 'relu'))
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
cnn.add(tf.keras.layers.Dropout(0.2))
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

#Training the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = cnn.fit(x = training_set, validation_data = test_set, epochs = 15)

#Plotting the training process
import matplotlib.pyplot as plt
plt.plot(history.history['loss'],label = 'Loss')
plt.plot(history.history['val_loss'],label = 'Val Loss')
plt.show()

#Plotting the training process
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],label = 'Accuracy')
plt.plot(history.history['val_accuracy'],label = 'Val Accuracy')
plt.show()

#Making a prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('hotdog/not_hotdog.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 0:
    prediction = 'hotdog'
else:
    prediction = 'not hotdog'
    
print(prediction)