import tkinter as tk
from tkinter import filedialog
from tkinter import *
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image,ImageTk
import numpy as np
import cv2

def FacialExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
    return model

top = tk.Tk()
top.geometry('800x600')
top.title('Emotion and Wrinkle Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font = ('arial',15,'bold'))
sign_image = Label(top)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model_a.json","model_weights.h5")
EMOTIONS_LIST = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

def Detect(file_path):
    global Label_packed
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image,scaleFactor = 1.05,minNeighbors=10)
    try:
        for(x,y,w,h) in faces:
            fc = gray_image[y:y+h,x:x+w]
            edges = cv2.Canny(fc,110,1000)        
            number_of_edges = np.count_nonzero(edges)
            roi = cv2.resize(fc,(48,48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
            print("Predicted Emotion is " + pred)
            if number_of_edges >= 1000:
                print("Wrinkle Found ")
                text=pred+" and Wrinkle Found"
            else:
                print("No Wrinkle Found ")
                text=pred+" and No Wrinkle Found"
            label1.configure(foreground="#011638",text = text)

    except:
        label1.configure(foreground = "#011638",text = "Unable to detect, Wrinkle can't be detected!...")


def show_Detect_button(file_path):
    detect_b = Button(top,text = "Detect Emotion and Wrinkles", command = lambda: Detect(file_path),padx =10,pady=5)
    detect_b.configure(background="#364156", foreground='white',font=('arial', 10, 'bold'))
    detect_b.place(relx=0.72, rely=0.46)


def upload_image():
    try:
        file_path= filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im= ImageTk.PhotoImage(uploaded)
        sign_image.configure(image =im)
        sign_image.image = im
        label1.configure(text = '')
        show_Detect_button(file_path)
    except:
        pass
    
upload = Button(top, text = "Upload Image", command=upload_image,padx=10,pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading =Label(top, text='Emotion and Wrinkle Detector',pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()
top.mainloop()