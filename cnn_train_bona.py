# Bonaventure Dossou - MSc in Data Engineering - Jacobs University Bremen
# HIDA Datathon - DLR_LCZ Challenge

# set the matplotlib backend so figures can be saved in the background
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from model_one_full import FSER20_1
import h5py
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
session = tf.Session(config=config)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

def generator(h5path, batchSize, num=None):

    db = h5py.File(h5path, "r")
    indices=np.arange(num)

    while True:

        np.random.shuffle(indices)
        for i in range(0, len(indices), batchSize):

            batch_indices = indices[i:i+batchSize]
            batch_indices.sort()

            by = db["label"][batch_indices,:]
            bx = db["sen2"][batch_indices,:,:,:]

            yield (bx,by)


'path to save models from check points:'
file0 = './'

train_file = '../dlr_challenge/data/training.h5'
testing_file = '../dlr_challenge/data/validation.h5'

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 20
INIT_LR = 1e-3
BS = 16
num_classes = 17
width = 32
height = 32
depth = 10

# number of all samples in training and validation sets'
trainNumber = 281893 # 80% of the whole dataset
# validationNumber = 24119
validationNumber = 70473 # 20% of the training dataset

model_1 = FSER20_1.build(width=width, height=height, depth=depth, classes=num_classes)

opt = SGD(INIT_LR, 0.9)
print("Model Created")
model_1.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])

modelbest = file0 + "_" + str(BS) +"_weights.best.hdf5"
checkpointer = ModelCheckpoint(modelbest, verbose=1, save_best_only=True, monitor='val_acc',
                               mode='max')

# As soon as the loss of the model begins to increase on the test dataset, we will stop training.
# First, we can define the early stopping callback.
# Simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# # H = model_1.fit(aug.flow(trainX, trainY, batch_size=BS), epochs=EPOCHS, verbose=1,
# #                           validation_data=(testX, testY), callbacks=[checkpointer, es],
# #                           steps_per_epoch=len(trainX) // BS)

H = model_1.fit_generator(generator(train_file, batchSize=BS, num=trainNumber),
                steps_per_epoch = trainNumber//BS,
                validation_data= generator(train_file, num=validationNumber, batchSize=BS),
                validation_steps = validationNumber//BS,
                epochs=EPOCHS,
                callbacks=[checkpointer, es])


plt.figure()
N = EPOCHS
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["acc"], label="train_acc")
plt.plot(H.history["val_acc"], label="val_acc")
plt.title("Training Log_loss/Accuracy")
plt.xlabel("Training Epochs")
plt.ylabel("Log_loss/Accuracy")
plt.legend(loc="best")
plt.savefig("training_plot.jpg")
plt.show()


def predata4LCZ(file, keyX, keyY):
    hf = h5py.File(file, 'r')
    x_tra = np.array(hf[keyX])
    y_tra = np.array(hf[keyY])
    hf.close()
    print(x_tra.shape, y_tra.shape)
    return x_tra, y_tra

'loading test data'
x_tst, y_tst= predata4LCZ(testing_file, 'sen2', 'label')

# 4. test phase
y_pre = model_1.predict(x_tst, batch_size=BS)
y_pre = y_pre.argmax(axis=-1)+1
y_testV = y_tst.argmax(axis=-1)+1

C = confusion_matrix(y_testV-1, y_pre-1, labels=np.arange(num_classes))
print('Confusion_matrix:')
print(C)

classRep = classification_report(y_testV, y_pre)
oa = accuracy_score(y_testV, y_pre)
cohKappa = cohen_kappa_score(y_testV, y_pre)

print('Classwise accuracy:')
print(classRep)

print('Overall accuracy and Kappa:')
print(oa, cohKappa)

prediction = model_1.predict(x_tst, verbose=1)
predicted_classes = np.argmax(prediction, axis=1)
labels_names = np.arange(num_classes)

# plt.style.use("dark_background")
cm = confusion_matrix(y_testV-1, y_pre-1,labels=np.arange(num_classes))
fig1, ax1 = plt.subplots(figsize=(10, 10))
sns.heatmap(cm.astype('int'), annot=True, fmt='.2f', xticklabels=labels_names, yticklabels=labels_names)
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
ax1.set_title("Numerical Confusion Matrix after {} epochs".format(EPOCHS))
plt.savefig("num_conf_matrix_2.png")
plt.show(block=False)

# plt.style.use("dark_background")
# Normalise
cmn = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels_names, yticklabels=labels_names)
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
ax.set_title("Normalized Confusion Matrix after {} epochs".format(EPOCHS))
plt.savefig("norm_conf_matrix_2.png")
plt.show(block=False)
