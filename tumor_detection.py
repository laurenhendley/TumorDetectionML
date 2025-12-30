import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn as skl
import matplotlib.pyplot as mpl
import cv2 as cv
from imutils import paths
import os 
import keras
import kagglehub

from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import layers

from keras.layers import Input, Dense, AveragePooling2D, Dropout, Flatten
from keras.applications import VGG16
from keras.models import Model
from keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix


# Download latest version
path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")

# Initial data parse
def eda(path):
    images = []
    labels = []

    image_path = list(paths.list_images(path))

    for ip in image_path:
        lbl = ip.split(os.path.sep)[-2]
        img = cv.imread(ip)
        img = cv.resize(img, (224,224))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        images.append(img)
        labels.append(lbl)

    return images, labels

# Plots the image
def plot_image(img):
    mpl.imshow(img)

# Noramalize images and labels as numpy arrays
def normalize(images , labels):
    images = np.array(images) / 255.0
    labels = np.array(labels)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    return images, labels, lb



#### Building the CNN (VGG16) ####

def CNN():
    train_generator = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1), layers.RandomZoom(0.1),layers.RandomTranslation(0.1,0.1)])

    inputs = Input(shape=(224,224,3))
    base_output = train_generator(inputs)

    base_model = VGG16(weights = 'imagenet', include_top = False)
    base_model.trainable = False
    base_output = base_model(base_output, training=False)

    base_output = AveragePooling2D(pool_size = (4,4)) (base_output)
    base_output = Flatten(name = "flatten") (base_output)
    base_output = Dense(64, activation = "relu") (base_output)
    base_output = Dropout(0.5) (base_output)

    outputs = Dense(2, activation="softmax")(base_output)

    model = Model(inputs, outputs)
    model.compile(optimizer = Adam(learning_rate  = 1e-3), metrics = ['accuracy'], loss = 'binary_crossentropy')

    return model


def train_model(model, train_ds, test_ds, epochs = 10):
    history = model.fit(
        train_ds,
        validation_data = test_ds,
        epochs = epochs
    )
    return history

def evaluate_model(model, test_X, batch_size, test_Y, label_binarizer):
    predictions = model.predict(test_X, batch_size = batch_size)
    predictions = np.argmax(predictions, axis = 1)

    actuals = np.argmax(test_Y, axis = 1)

    print(classification_report(actuals, predictions, target_names=label_binarizer.classes_))
    cm = confusion_matrix(actuals, predictions)

    return cm


def accuracy(cm):
    total = sum(sum(cm))
    accuracy = (cm[0,0]+cm[1,1]) / total
    print("Accuracy: {:.4f}".format(accuracy))


def plot_metrics(epochs,history):
    N = epochs
    mpl.style.use("ggplot")
    mpl.figure()

    mpl.plot(np.arrange(0, N), history.history["loss"], label = "train_loss")
    mpl.plot(np.arrange(0, N), history.history["val_loss"], label = "val_loss")
    mpl.plot(np.arrange(0, N), history.history["accuracy"], label = "accuracy")
    mpl.plot(np.arrange(0, N), history.history["val_accuracy"], label = "val_accuracy")

    mpl.title("Training loss and accuracy on brain tumor dataset")
    mpl.xlabel("Epoch")
    mpl.ylabel("Loss/accuracy")
    mpl.legend(loc="lower left")
    mpl.savefig("brainTumorPlot.jpg")


# Main function
if __name__ == '__main__':
    images, labels = eda(path)
    images, labels, lb = normalize(images, labels)

    (train_X, test_X, train_Y, test_Y) = train_test_split(images, labels, test_size = 0.1, random_state = 42, stratify = labels)

    train_ds = tf.data.Dataset.from_tensor_slices((train_X,train_Y)).shuffle(500).batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((test_X,test_Y)).batch(32).prefetch(tf.data.AUTOTUNE)


    model = CNN()
    model.summary()

    history = train_model(model, train_ds, test_ds, epochs = 10)

    cm = evaluate_model(model, test_X, 32, test_Y, lb)

    accuracy(cm)

    plot_metrics(epochs = 10, history = history)
