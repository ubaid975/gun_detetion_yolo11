
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import PIL
import gradio
gun_detect=YOLO("best.pt")
def detect(img):
  res=gun_detect(img)
  res_plotted=res[0].plot()
  return PIL.Image.fromarray(res_plotted)

app=gradio.Interface(fn=detect,inputs=['image'],outputs="image")
app.launch()

