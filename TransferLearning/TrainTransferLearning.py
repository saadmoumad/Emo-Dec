import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import TransferModels



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output loss/accuracy plot")
ap.add_argument("-t", "--transfer", type=str, default="inception_v3",
    help="Transfer Learning base model")
ap.add_argument("-m", "--model", type=str,
    default="transfer.model",
    help="path to output face mask detector model")
args = vars(ap.parse_args())


# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32


print("[INFO] loading images...")
Train_imagePaths = list(paths.list_images('images/train'))
Train_data = []
Train_labels = []

Test_imagePaths = list(paths.list_images('images/test'))
Test_data = []
Test_labels = []

for imagePath in Train_imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the input image (48x48) and preprocess it
    image = load_img(imagePath, target_size=(48, 48))
    image = img_to_array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = preprocess_input(image)

    Train_labels.append(label)
    Train_data.append(image)

for imagePath in Test_imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the input image (48x48) and preprocess it
    image = load_img(imagePath, target_size=(48, 48))
    image = img_to_array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = preprocess_input(image)

    Test_labels.append(label)
    Test_data.append(image)

# convert the data and labels to NumPy arrays
Train_X = np.array(Train_data, dtype="float32")
Train_Y = np.array(Train_labels)

Test_X = np.array(Test_data, dtype="float32")
Test_Y = np.array(Test_labels)

# perform one-hot encoding on the labels
lb = LabelEncoder()

Train_Y = lb.fit_transform(Train_Y)
Train_Y = to_categorical(Train_Y)

Test_Y = lb.fit_transform(Test_Y)
Test_Y = to_categorical(Test_Y)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

#model
model = TransferModels.TransferLearningNN('inception_v3').model

# train the head of the network
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
    target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save(args["transfer"]+args["model"], save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])