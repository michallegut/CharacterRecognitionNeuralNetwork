import numpy as np
import pickle as pkl
import random
import time

np.seterr(all='ignore')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1.0 - y)

class MLP_Classifier(object):
    def __init__(self, input, hidden, output, iterations=100, learning_rate=0.01,
                 l2_in=0.0005, l2_out=0.0005, momentum=0, rate_decay=0, verbose=True):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l2_in = l2_in
        self.l2_out = l2_out
        self.momentum = momentum
        self.rate_decay = rate_decay
        self.verbose = verbose
        self.input = input + 1
        self.hidden = hidden
        self.output = output
        self.ai = np.ones(self.input)
        self.ah = np.ones(self.hidden)
        self.ao = np.ones(self.output)
        input_range = 1.0 / self.input ** (1 / 2)
        self.wi = np.random.normal(loc=0, scale=input_range, size=(self.input, self.hidden))
        self.wo = np.random.uniform(size=(self.hidden, self.output)) / np.sqrt(self.hidden)
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):
        if len(inputs) != self.input - 1:
            raise ValueError('Wrong number of inputs!')
        self.ai[0:self.input - 1] = inputs
        sum = np.dot(self.wi.T, self.ai)
        self.ah = sigmoid(sum)
        sum = np.dot(self.wo.T, self.ah)
        self.ao = sigmoid(sum)
        return self.ao

    def backPropagate(self, targets):
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets!')
        output_deltas = dsigmoid(self.ao) * -(targets - self.ao)
        error = np.dot(self.wo, output_deltas)
        hidden_deltas = dsigmoid(self.ah) * error
        change = output_deltas * np.reshape(self.ah, (self.ah.shape[0], 1))
        regularization = self.l2_out * self.wo
        self.wo -= self.learning_rate * (change + regularization) + self.co * self.momentum
        self.co = change
        change = hidden_deltas * np.reshape(self.ai, (self.ai.shape[0], 1))
        regularization = self.l2_in * self.wi
        self.wi -= self.learning_rate * (change + regularization) + self.ci * self.momentum
        self.ci = change
        error = sum(0.5 * (targets - self.ao) ** 2)
        return error

    def fit(self, patterns):
        num_example = np.shape(patterns)[0]
        for i in range(self.iterations):
            error = 0.0
            random.shuffle(patterns)
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error += self.backPropagate(targets)
            if self.verbose == True:
                error = error / num_example
                print("Time: {:.2f} minutes".format((time.time() - start) / 60))
                print("Iteration: {:d}".format(i + 1))
                print('Training error: %-.5f' % error)
                print()
            self.learning_rate = self.learning_rate * (
            self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))
            pkl.dump(NN.wi, open('inputWeights.pkl', mode='wb'), pkl.HIGHEST_PROTOCOL)
            pkl.dump(NN.wo, open('outputWeights.pkl', mode='wb'), pkl.HIGHEST_PROTOCOL)

    def predict(self, X):
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p.tolist()))
        return predictions

if __name__ == '__main__':
    data = pkl.load(open('train.pkl', mode='rb'))
    X = data[0]
    Y = data[1]
    Y_one_hot = np.zeros([31798, 36], int)
    for i in range(Y_one_hot.shape[0]):
        Y_one_hot[i][Y[i]] = 1
    patterns = []
    for i in range(X.shape[0]):
        patterns.append((X[i], Y_one_hot[i]))
    NN = MLP_Classifier(3136, 200, 36, iterations=100, learning_rate=0.01,
                        momentum=0.5, rate_decay=0.001)
    NN.wi = pkl.load(open('inputWeights.pkl', mode='rb'))
    NN.wo = pkl.load(open('outputWeights.pkl', mode='rb'))
    start = time.time()
    NN.fit(patterns)