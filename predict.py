#para compilar
#python3 predict.py -i imagem.png -m modelo.model
import cv2
from skimage.color import rgb2gray
import keras
import tensorflow as tf
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image we are going to classify")
ap.add_argument("-m", "--model", required=True,
	help="path to trained Keras model")
args = vars(ap.parse_args())

dici = {0:"INFECTADO", 1:"SAUDAVEL"}
data = []
#trabalha a imagem
image = rgb2gray(cv2.imread(args["image"]))
image = cv2.resize(image, (28, 28)).flatten()

data.append(image)
data = np.array(data)
data = data.reshape(1,28,28,1)
#pega o modelo
model = tf.keras.models.load_model(args["model"])
#preve
preds = model.predict(data)
i = preds.argmax(axis=1)[0]
label = dici[i]
print(label)