import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Dataset Loading
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "D:\\Core1DataSet\\Core1DataSet\\train",
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    label_mode='binary',
    class_names=['negative', 'positive'])

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    "D:\\Core1DataSet\\Core1DataSet\\validation",
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    label_mode='binary',
    class_names=['negative', 'positive'])

# Architecture
model = models.Sequential()

# Add 10 convolutional layers/8 currently working
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(1024, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(2048, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

# Flattening the output of layers
model.add(layers.Flatten())

# Dense Layers
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compilation with binary crossentropy loss
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training
history = model.fit(train_data,
                    validation_data=val_data,
                    epochs=10)

# Evaluation
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    'D:\\Core1DataSet\\Core1DataSet\\test',
    image_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    label_mode='binary',
    class_names=['negative', 'positive'])

test_loss, test_acc = model.evaluate(test_data)

# Printing the validation and test loss
print('Validation Loss:', history.history['val_loss'][-1])
print('Test Loss:', test_loss)

# Plotting
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')