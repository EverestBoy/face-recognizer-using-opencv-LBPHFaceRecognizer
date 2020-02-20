import cv2 
import requests
import cv2
import numpy as np
import os
import pickle


face_cascade = cv2.CascadeClassifier('cascades/face.xml')
eye_cascade =  cv2.CascadeClassifier('cascades/eye.xml')
url = 'http://192.168.1.115:8080/shot.jpg'

print("Urls is "+url)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"persons_name": 1}


with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}


while True:

	img_receive = requests.get(url)
	imgNp = np.array(bytearray(img_receive.content), dtype=np.uint8)
	grays = cv2.imdecode(imgNp, -1)
	gray = cv2.cvtColor(grays, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	
	
	for (x, y, w, h) in faces:

		roi_gray = gray[y:y+h, x:x+w]
		roi_color = gray[y:y+h, x:x+w]

		# if count < 8:
		# 	img_item = str(count)+".png"
		# 	cv2.imwrite(img_item, roi_gray)
		# 	count += 1

		id_, conf = recognizer.predict(roi_gray)
		if conf >= 75 and conf <=125: #and conf <= 85:
			print(conf)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255,0,0,0)
			stroke = 2
			cv2.putText(grays, name,(x,y), font, 1, (255,255,255))

		# print(x,y,w,h) 
		color = (255,0,0,0)
		stroke = 2
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(grays, (x, y), (end_cord_x, end_cord_y), color, stroke)
		# nose = nose_cascade.detectMultiScale(roi_gray)
		eye = eye_cascade.detectMultiScale(roi_gray)
		
		for (exe,eye,ewe,ehe) in eye:
			cv2.rectangle(roi_color,(exe,eye),(exe+ewe,eye+ehe),(0,255,0),2)

	
	
	cv2.imshow('image',grays)
	
	
	

	if(ord('q') == cv2.waitKey(10)):
		exit(0)
