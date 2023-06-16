from keras.models import load_model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf


# generates the confusion matrix of the trained model

model = load_model('C:/Users/eduar/Desktop/Bachelors Thesis/Trained Model/Trained')

testX = np.load('testX.npy')
testY = np.load('testY.npy')

predictions = model.predict(testX)
results = np.argmax(predictions, axis=1)
roundedLabels = np.argmax(testY, axis=1)
confusionMatrix = confusion_matrix(roundedLabels, results)
sns.set(rc={'figure.figsize':(25, 25)})
f = sns.heatmap(confusionMatrix, annot=True, annot_kws={"size": 20}, fmt='g')
plt.show()

with np.printoptions(threshold=np.inf):
    with open('confusionMatrix.txt', 'w') as file:
        file.write(str(confusionMatrix))

