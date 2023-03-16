from keras import models
import face_recognition
import os
import cv2
import numpy as np

font = cv2.FONT_HERSHEY_DUPLEX
input_shape = (48,48)
emocoes =[
           'Raiva',
           'Nojo',
           'Medo',
           'Alegria',
           'Tristeza',
           'Surpresa',
           'Neutro'
          ]
modelo = models.load_model(os.getcwd()+"\emotions_model")

def preprocessa_imagem(imagem, modelo):
  imagem = imagem.astype(np.float64)/255
  imagem = np.expand_dims(imagem, axis = 0)
  probabilidades = modelo.predict(imagem)
  probabilidades = probabilidades[0]
  classificacao = np.argmax(probabilidades)
  certeza = round(probabilidades[np.argmax(probabilidades)]*100, ndigits= 2)

  return classificacao, certeza, probabilidades


def add_box_to_frame(frame):
    # frame = np.zeros([480, 640, 4], dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_recognition.face_locations(gray)
    for (top, right, bottom, left) in faces:
        img_predict = cv2.resize(gray[top:bottom, left:right],input_shape)
        resultado = preprocessa_imagem(img_predict,modelo)
        frame = cv2.rectangle(frame, (left-50, top-50), (right+50, bottom+50), (255, 0, 0), 2)
        frame = cv2.rectangle(frame,(left-50, bottom + 15 ), (right+50, bottom+50), (255, 0, 0), cv2.FILLED)
        frame = cv2.putText(frame,emocoes[resultado[0]], (left, bottom +40), font, 1.0, (255, 255, 255), 1)

    # frame[:, :, 3] = (frame.max(axis=2) > 0).astype(int) * 255

    return frame

