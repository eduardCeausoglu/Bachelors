import os
import pandas as pd
from scipy.misc import imread
import numpy as np
import cv2
import keras
import seaborn as sns
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential
import matplotlib.pyplot as plt


class Model:
    def __init__(self, dataPath, modelPath):
        self.dataPath = os.path.abspath(dataPath)
        self.trainedModelPath = os.path.abspath(modelPath)
        self.trainData = []
        self.trainTargetAttributes = []
        self.testData = []
        self.testTargetAttributes = []
        self.validationData = []
        self.validationTargetAttributes = []


    def resizeCv(self, image):
        """

        :param image: numpy.ndarray
        :return: numpy.ndarray
        """
        return cv2.resize(image, (30, 30), interpolation=cv2.INTER_LINEAR)

    def loadData(self):
        """
        loads data located at the path stored in self.dataPath
        saves it in self.trainData and self.trainTargetAttributes
        :return: void
        """
        for directory in os.listdir(self.dataPath):
            innerDir = os.path.join(self.dataPath, directory)
            csvFile = pd.read_csv(os.path.join(innerDir, 'GT-' + directory + '.csv'), sep=';')
            for row in csvFile.iterrows():
                imgPath = os.path.join(innerDir, row[1].Filename)
                img = imread(imgPath)
                img = img[row[1]['Roi.X1']:row[1]['Roi.X2'], row[1]['Roi.Y1']:row[1]['Roi.Y2'], :]
                img = self.resizeCv(img)
                self.trainData.append(img)
                self.trainTargetAttributes.append(row[1].ClassId)

    def randomizeData(self):
        """
        randomizes data and splits it into train, test, and validation sets
        these are stored in self.trainData, self.testData, self.validationData,
        self.trainTargetAttributes, self.testTargetAttributes, self.validationTargetAttributes
        :return: void
        """
        trX = np.stack(self.trainData)
        trY = keras.utils.np_utils.to_categorical(self.trainTargetAttributes)
        randomize = np.arange(len(trX))
        np.random.shuffle(randomize)
        x = trX[randomize]
        y = trY[randomize]

        split = int(x.shape[0] * 0.6)
        trainX, validationX = x[:split], x[split:]
        trainY, validationY = y[:split], y[split:]

        split = int(validationX.shape[0] * 0.5)
        validationX, testX = validationX[:split], validationX[split:]
        validationY, testY = validationY[:split], validationY[split:]

        self.trainData = trainX
        self.trainTargetAttributes = trainY

        self.testData = testX
        self.testTargetAttributes = testY

        self.validationData = validationX
        self.validationTargetAttributes = validationY

    def saveDistribution(self):
        """
        saves the distributions in .npy files
        :return: void
        """
        np.save('trainX.npy', self.trainData)
        np.save('trainY.npy', self.trainTargetAttributes)
        np.save('testX.npy', self.testData)
        np.save('testY.npy', self.testTargetAttributes)
        np.save('validationX.npy', self.validationData)
        np.save('validationY.npy', self.validationTargetAttributes)

    def buildAndTrainModel(self):
        """
        builds, compiles and trains the model
        :return: void
        """
        hiddenNumUnits = 2048
        hiddenNumUnits1 = 1024
        hiddenNumUnits2 = 128
        outputNumUnits = 43

        epochs = 7
        batch_size = 16
        pool_size = (2, 2)

        self.model = Sequential([

            Conv2D(16, (3, 3), activation='relu', input_shape=(30, 30, 3), padding='same'),
            BatchNormalization(),

            Conv2D(16, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=pool_size),
            Dropout(0.2),

            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),

            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=pool_size),
            Dropout(0.2),

            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),

            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=pool_size),
            Dropout(0.2),

            Flatten(),

            Dense(units=hiddenNumUnits, activation='relu'),
            Dropout(0.3),
            Dense(units=hiddenNumUnits1, activation='relu'),
            Dropout(0.3),
            Dense(units=hiddenNumUnits2, activation='relu'),
            Dropout(0.3),
            Dense(units=outputNumUnits, input_dim=hiddenNumUnits, activation='softmax'),
        ])

        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
        trainedModel = self.model.fit(self.trainData.reshape(-1, 30, 30, 3), self.trainTargetAttributes, epochs=epochs, batch_size=batch_size,
                                       validation_data=(self.validationData, self.validationTargetAttributes))

        print("Accuracy: " + str(trainedModel.history))

    def saveModel(self):
        """
        saves the trained model at the path stored in self.trainedModelPath
        :return: void
        """
        self.model.save(self.trainedModelPath)

    def plotDataset(self):
        """
        plots the traffic sign frequency diagram
        :return: void
        """
        fig = sns.distplot(self.trainTargetAttributes, color='purple', kde=False, bins=43, hist=True,
                           hist_kws=dict(edgecolor="black", linewidth=3))
        fig.set(title="Traffic sign frequency by class id",
                xlabel="Traffic sign Class ID",
                ylabel="Frequency")
        plt.show()

