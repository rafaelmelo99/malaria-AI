from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--database", required=True,
	help="path to database")
ap.add_argument("-m", "--model", required=True,
	help="path to save the model")
args = vars(ap.parse_args())

data = []
labels = []

print("[INFO] Loading images...")
# pega as imagens e as embaralha
imagePaths = sorted(list(paths.list_images(args["database"])))
random.seed(42)
random.shuffle(imagePaths)
 
for imagePath in imagePaths:
	image = rgb2gray(cv2.imread(imagePath))
	image = cv2.resize(image, (28, 28)).flatten()
	data.append(image)
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

data = np.array(data)
labels = np.array(labels)

#separando o treino do teste
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=42)
x_train.shape
x_test.shape

x_train = x_train.reshape(20668,28,28,1)
x_test = x_test.reshape(6890,28,28,1)

from keras.utils import to_categorical
#categorizando as labels
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)


#criando o modelo
print("[INFO] Training network...")
model = keras.Sequential([
    keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)),
    keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(2, activation='softmax')
])

#compilando
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#treinando
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

#salvando o modelo no lugar passado
model.save(args["model"],overwrite=True)