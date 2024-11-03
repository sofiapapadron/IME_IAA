#Script que utiliza opencv para probar la red neuronal convolucional

import cv2
import numpy as np
import os
import sys
from keras.models import load_model
from keras.preprocessing import image

def preprocess_image(img):
    #Obtenemos el umbral y el numero de iteraciones
    thresh_val = cv2.getTrackbarPos('Umbral', 'frame')
    iter_val = cv2.getTrackbarPos('Iteraciones', 'frame')
    
    #Preprocesamiento de la imagen
    frame = img.copy() #Copiamos la imagen para no modificar la original
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convertimos a escala de grises

    ret, thresh_img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV) #Aplicamos umbralizacion para obtener una imagen binaria

    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=iter_val) #Aplicamos operaciones morfologicas para eliminar ruido

    thresh_img = cv2.resize(thresh_img, (28, 28), interpolation=cv2.INTER_LINEAR) #Redimensionamos a 28x28

    thresh_img = np.reshape(thresh_img, (1, 28, 28, 1)) #Agregamos una dimension para que sea compatible con la entrada de la red neuronal

    return thresh_img

def get_prediction(img, model):
    #Obtenemos la prediccion de la red neuronal
    prediction = model.predict(img)

    #Obtenemos el valor maximo de la prediccion
    #En este caso, el indice corresponde al digito que se esta mostrando
    return np.argmax(prediction)

def show_image(img, pred_val):
    #Redimensionamos la imagen para que se vea mejor
    final_img = img.copy()
    final_img = final_img.reshape((28, 28)) #Redimensionamos a 28x28
    final_img = cv2.resize(final_img, (224, 224), interpolation=cv2.INTER_LINEAR)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)

    if pred_val: 
        print("Prediccion: {}".format(pred_val))
        cv2.putText(final_img, str(pred_val), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

     #Mostramos la imagen
    cv2.imshow('frame', final_img)

#Cargamos el modelo
model = load_model('./mnist_cnn.h5')

#Captura de video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se puede abrir la camara")
    exit()


cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.createTrackbar('Umbral', 'frame', 0, 255, lambda x: x)
cv2.createTrackbar('Iteraciones', 'frame', 1, 7, lambda x: x)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while cv2.waitKey(1) != ord('q'):
    ret, frame = cap.read()

    if not ret:
        print("No se puede recibir el frame. Finalizando ...")
        break
    
    gray = preprocess_image(frame) #Preprocesamos la imagen

    #Clasificacion
    prediction = get_prediction(gray, model)

    show_image(gray, prediction) #Mostramos la imagen

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()