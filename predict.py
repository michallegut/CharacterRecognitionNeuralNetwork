import numpy as np
import pickle as pkl

class NeuralNetwork(object):
    def __init__(self, inputNeurons, hiddenNeurons, outputNeurons):
        self.inputNeurons = inputNeurons + 1
        self.hiddenNeurons = hiddenNeurons
        self.outputNeurons = outputNeurons
        self.inputActivation = np.ones(self.inputNeurons)
        self.hiddenActivation = np.ones(self.hiddenNeurons)
        self.outputActivation = np.ones(self.outputNeurons)
        self.inputWeights = pkl.load(open('inputWeights.pkl', mode = 'rb'))
        self.outputWeights = pkl.load(open('outputWeights.pkl', mode = 'rb'))

    def feedForward(self, input):
        self.inputActivation[0:self.inputNeurons - 1] = input
        sum = np.dot(self.inputWeights.T, self.inputActivation)
        self.hiddenActivation = 1 / (1 + np.exp(-sum))
        sum = np.dot(self.outputWeights.T, self.hiddenActivation)
        self.outputActivation = 1 / (1 + np.exp(-sum))
        return self.outputActivation

def predict(x):
    neuralNetwork = NeuralNetwork(3136, 200, 36)
    predictions = np.empty([x.shape[0], 1])
    for i in range (x.shape[0]):
        predictions[i][0] = np.argmax(neuralNetwork.feedForward(x[i]))
    return predictions