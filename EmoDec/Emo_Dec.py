import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imutils import paths
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import argparse
import imutils
import time
import os


# construct the argument parser 
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="data",
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--mode", type=str, required=True,
	help="Init, Train or Stream. By default streaming")
ap.add_argument("-w", "--weights", type=str,
	default="mymodel.h5",
	help="model's weights file")
ap.add_argument("-f", "--face", type=str,
	default="haarcascade_frontalface",
	help="face detection model. option: haarcascade_frontalface/res10")
ap.add_argument("-c", "--confidence", type=float,
	default=0.5 ,
	help="Level of confidence for face detections")
args = vars(ap.parse_args())

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
confidence = args["confidence"]

train_dir = 'Data/'+args['dataset']+'/train'
val_dir = 'Data/'+args['dataset']+'/test'

num_train = len(list(paths.list_images(train_dir)))
num_val = len(list(paths.list_images(val_dir)))
batch_size = 64
num_epoch = 50

checkpoint_dir = './training_checkpoints'

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
				train_dir,
				target_size=(48,48),
				batch_size=batch_size,
				color_mode="grayscale",
				class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
				val_dir,
				target_size=(48,48),
				batch_size=batch_size,
				color_mode="grayscale",
				class_mode='categorical')

def model():
		model = Sequential()

		model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
		model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(1024, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(7, activation='softmax'))

		return model

def Res10_function(Frame, FaceModel, EmotionsModel):
	global confidence

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	FaceModel.setInput(blob)
	detections = FaceModel.forward()

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		
			confidence = detections[0, 0, i, 2]

			if confidence > 0.5 :
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				face = frame[startY:endY, startX:endX]
				face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
				cropped_img = np.expand_dims(np.expand_dims(cv2.resize(face, (48, 48)), -1), 0)               
				
				faces.append(cropped_img)
				locs.append((startX, startY, endX, endY))
			else :
					break

	if len(faces) > 0:
		preds = EmotionsModel.predict(faces)

	return (locs, preds)

# plots accuracy and loss curves
def plot_model_history(model_history):
		fig, axs = plt.subplots(1,2,figsize=(15,5))
		# summarize history for accuracy
		axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
		axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
		axs[0].set_title('Model Accuracy')
		axs[0].set_ylabel('Accuracy')
		axs[0].set_xlabel('Epoch')
		axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
		axs[0].legend(['train', 'val'], loc='best')
		# summarize history for loss
		axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
		axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
		axs[1].set_title('Model Loss')
		axs[1].set_ylabel('Loss')
		axs[1].set_xlabel('Epoch')
		axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
		axs[1].legend(['train', 'val'], loc='best')
		fig.savefig('plot.png')
		plt.show()



if args['mode'] == "Init":
		model = model()

		model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])

		#Checkpoints for model training
		checkpoint_prefix = os.path.join(checkpoint_dir, args['dataset']+"ckpt_{epoch}")

		checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_prefix,
			save_weights_only=True)

		model_info = model.fit_generator(
					train_generator,
					steps_per_epoch=num_train // batch_size,
					epochs=num_epoch,
					validation_data=validation_generator,
					validation_steps=num_val // batch_size,
					callbacks=[checkpoint_callback])

		plot_model_history(model_info)

		fer_json = model.to_json()
		with open("mymodel.json", "w") as json_file:
			json_file.write(fer_json)

		model.save_weights('mymodel.h5')

elif args['mode'] == "Train":
		model = model()
		model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

		#Checkpoints for model training
		checkpoint_prefix = os.path.join(checkpoint_dir, args['dataset']+"ckpt_{epoch}")

		checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_prefix,
			save_weights_only=True)

		model_info = model.fit_generator(
					train_generator,
					steps_per_epoch=num_train // batch_size,
					epochs=num_epoch,
					validation_data=validation_generator,
					validation_steps=num_val // batch_size,
					callbacks=[checkpoint_callback])

		plot_model_history(model_info)

else:
		model = model()
		model.load_weights('mymodel.h5')

		print("[INFO] starting video stream...")
		vs = VideoStream(src=0).start()
		time.sleep(2.0)
		print("[INFO] video stream started successfully.")

		if args['face'] == "haarcascade_frontalface":

				print("[INFO] loading face detection model...")
				facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
				print("[INFO] using haarcascade_frontalface face detection model")
				while True:
						
						frame = vs.read()
						frame = imutils.resize(frame, width=1200)

						#print("[INFO] loading face detection model...")
						#facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
						gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
						faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
						#print("[INFO] using haarcascade_frontalface face detection model")
						
						for (x, y, w, h) in faces:
							cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
							roi_gray = gray[y:y + h, x:x + w]
							cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
							prediction = model.predict(cropped_img)
							maxindex = int(np.argmax(prediction))
							cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

						cv2.imshow("Frame", frame)
						if cv2.waitKey(1) & 0xFF == ord('q'):
							break

		else :
			print("[INFO] loading face detection model...")
			prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
			weightsPath = os.path.sep.join(["face_detector",
				"res10_300x300_ssd_iter_140000.caffemodel"])
			FaceModel = cv2.dnn.readNet(prototxtPath, weightsPath)
			time.sleep(2.0)
			print("[INFO] using res10 face detection model")

			while True:
				frame = vs.read()
				frame = imutils.resize(frame, width=1200)

				(locs, preds) = Res10_function(frame, FaceModel, model)

				for (box, pred) in zip(locs, preds):

						(startX, startY, endX, endY) = box
						
						cv2.putText(frame, emotion_dict[int(np.argmax(pred))], (startX, startY - 10),
								cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
						cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)


				cv2.imshow("Frame", frame)
				key = cv2.waitKey(1) & 0xFF

				if key == ord("q"):
					break

		print("[INFO] stopping video stream...")
		cv2.destroyAllWindows()
		vs.stop()
