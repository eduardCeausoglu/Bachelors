from Application.Model import Model
from Application.Server import Server
from flask import Flask
import os

class Controller:
    def __init__(self, dataPath, trainedModelPath):
        self.dataPath = dataPath
        self.trainedModelPath = trainedModelPath
        self.model = self.generateModel()
        self.app, self.server = self.generateServer()


    def generateModel(self):
        """
        generates an object of type Model, with the paths provided
        :return: Model
        """
        return Model(self.dataPath, self.trainedModelPath)

    def generateServer(self):
        """
        generates an object of type Server, and an object of type App
        :return: App, Server
        """
        app = Flask(__name__)
        server = Server.register(app, route_base='/')
        return app, server
        
    def trainModel(self):
        """
        calls the necessary functions in the model in order to plot the data distribution graph,
        test, generate, train, and save a model and the associated data (data distribution files)
        :return: Void
        """
        self.model.loadData()
        self.model.plotDataset()
        self.model.randomizeData()
        self.model.saveDistribution()
        self.model.buildAndTrainModel()
        self.model.saveModel()

    def runServer(self):
        """
        runs the server app
        :return: Void
        """
        self.app.run(debug=True)

    def testModel(self):
        """
        executes assertions to test the functionality of the Model class
        :return: Void
        """

        # testing the paths
        path = "C:/Users/eduar/Desktop/Bachelors Thesis/Test"
        modelPath = "C:/Users/eduar/Desktop/Bachelors Thesis/Test/Trained"
        testModel = Model(self.dataPath, modelPath)
        assert(testModel.dataPath == self.dataPath.replace('/', '\\'))
        assert(testModel.trainedModelPath == modelPath.replace('/', '\\'))

        # testing the data distribution in training / test / validation sets
        testModel.loadData()
        assert(len(testModel.trainData) == 39209) # check if all the data was loaded properly
        testModel.randomizeData()
        assert(len(testModel.trainData) == 23525) # 60% of the dataset
        assert(len(testModel.testData) == 7842) # 20% of the dataset
        assert(len(testModel.validationData) == 7842) # 20% of the dataset

        # testing the saving of the model
        assert(len(os.listdir(path)) == 0) # no model has been saved yet
        testModel.buildAndTrainModel()
        testModel.saveModel()
        assert(len(os.listdir(path)) == 1) # the model should be saved now