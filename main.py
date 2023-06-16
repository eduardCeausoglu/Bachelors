from Application.Controller import Controller

# define the path for loading the data and saving the model
dataPath = 'C:/Users/eduar/Desktop/Bachelors Thesis/GTSRB/Final_Training/Images'
trainedModelPath = 'C:/Users/eduar/Desktop/Bachelors Thesis/Trained Model/Trained'

# instantiate the controller
controller = Controller(dataPath, trainedModelPath)

# call the corresponding functions in the model in order to:
# test the model  - #1
# train the model - #2
# run the server  - #3

#controller.testModel()     #1
#controller.trainModel()    #2
controller.runServer()     #3

