from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from model import L1SDist
from data_processing import preprocess
import cv2
import tensorflow as tf
import os
import numpy as np
import dlib

class CamApp(App):
    def build(self):
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        self.model = tf.keras.models.load_model('./model/SNN.keras', custom_objects = {'L1SDist': L1SDist, 
                                                                               'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy})
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def update(self, *args):
        ret, frame = self.capture.read()
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def verify(self, *args):
        detector = dlib.get_frontal_face_detector()
        det_threshold = 0.9
        ver_threshold = 0.8
        SAVE_PATH = './realtime_data/input/input_image.jpg'
        ret, frame = self.capture.read()
        faces = detector(frame)
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_img = frame[y+10:y+h+10, x+10:x+w+10]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite(SAVE_PATH, face_img)
        predictions = []
        for image in os.listdir('./realtime_data/verification'):
            input_img = preprocess('./realtime_data/input/input_image.jpg')
            if image != '.DS_Store':
                validation_img = preprocess(f'./realtime_data/verification/{image}')
                prediction = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
                predictions.append(prediction)  
        detection = 0     
        for pred in predictions:
            if pred >= det_threshold:
                detection += 1 
        verification = detection / (len(os.listdir('./realtime_data/verification')) - 1)
        verified = verification >= ver_threshold
        self.verification_label.text = 'Nikita' if verified == True else 'Unknown'

        Logger.info(predictions)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        
        return predictions, verified



if __name__ == '__main__':
    CamApp().run()