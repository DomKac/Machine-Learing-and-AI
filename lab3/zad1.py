import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


def relu(x):
        return x * (x > 0)


def relu_derivative(x):
    return 1. * (x > 0)


class NeuralNetwork381:
    def __init__(self, x, y):
        self.input      = x.T
        self.weights1   = np.random.rand(self.input.shape[0], 8).T
        self.weights2   = np.random.rand(8, 1).T
        self.y          = y.T
        self.output     = np.zeros(self.y.shape)
        self.eta        = 0.5 # learning rate
        

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.weights1, self.input))
        self.output = sigmoid(np.dot(self.weights2, self.layer1))


    def backprop(self):
        delta2 = (self.y - self.output) * sigmoid_derivative(self.output)
        d_weights2 = self.eta * np.dot(delta2, self.layer1.T)
        
        delta1 = np.dot(self.weights2.T, delta2) * sigmoid_derivative(self.layer1)
        d_weights1 = self.eta * np.dot(delta1, self.input.T)
        
        self.weights1 += d_weights1
        self.weights2 += d_weights2


    def feedforward_relu(self):
        self.layer1 = relu(np.dot(self.weights1, self.input))
        self.output = sigmoid(np.dot(self.weights2, self.layer1))

    def backprop_relu(self):
        delta2 = (self.y - self.output) * sigmoid_derivative(self.output)
        d_weights2 = self.eta * np.dot(delta2, self.layer1.T)
        
        delta1 = np.dot(self.weights2.T, delta2) * relu_derivative(self.layer1)
        d_weights1 = self.eta * np.dot(delta1, self.input.T)
        
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def solve(self, x):
        layer1 = sigmoid(np.dot(self.weights1, x.T))
        output = sigmoid(np.dot(self.weights2, layer1))        
        return output
    

class NeuralNetwork3841:
    def __init__(self, x, y):
        self.input      = x.T
        self.weights1   = np.random.rand(self.input.shape[0], 8).T
        self.weights2   = np.random.rand(8, 4).T
        self.weights3   = np.random.rand(4, 1).T
        self.y          = y.T
        self.output     = np.zeros(self.y.shape)
        self.eta        = 0.5 # learning rate
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.weights1, self.input))
        self.layer2 = sigmoid(np.dot(self.weights2, self.layer1))
        self.output = sigmoid(np.dot(self.weights3, self.layer2))

    def backprop(self):
        delta3 = (self.y - self.output) * sigmoid_derivative(self.output)
        d_weights3 = self.eta * np.dot(delta3, self.layer2.T)

        delta2 = np.dot(self.weights3.T, delta3) * sigmoid_derivative(self.layer2)
        d_weights2 = self.eta * np.dot(delta2, self.layer1.T)

        delta1 = np.dot(self.weights2.T, delta2) * sigmoid_derivative(self.layer1)
        d_weights1 = self.eta * np.dot(delta1, self.input.T)
        
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights3 += d_weights3

class NeuralNetwork:
    def __init__(self, x, y, hidden_layer_sizes):
        self.layer_sizes = [x.shape[1]] + hidden_layer_sizes + [y.shape[1]] # lista z ilościami neuronów we wszystkich warstwach
        self.layers_num  = len(self.layer_sizes)    # liczba wszytskich warstw
        self.layer       = [None] * self.layers_num # lista warstw z neuronami
        self.weights     = [np.random.rand(self.layer_sizes[i - 1], self.layer_sizes[i]).T for i in range(1, self.layers_num)]

        self.layer[0]    = x.T
        self.y           = y.T
        self.eta         = 0.5 # learning rate


    def feedforward(self):
        for i in range(1, self.layers_num):
            self.layer[i] = sigmoid(np.dot(self.weights[i - 1], self.layer[i - 1])) # (self.weights[i - 1] @ self.layer[i - 1]) 

    def backprop(self):
        delta_m = (self.y - self.layer[-1]) * sigmoid_derivative(self.layer[-1])
        for m in range(self.layers_num - 1, 0, -1):
            delta_m_1 = np.dot(self.weights[m - 1].T, delta_m) * sigmoid_derivative(self.layer[m - 1])
            d_weights = self.eta * np.dot(delta_m, self.layer[m - 1].T)
            self.weights[m - 1] += d_weights
            delta_m = delta_m_1
        self.output = self.layer[-1]


# ---------------------- TESTY ----------------------



def test_381():
    print('-'*50, 'Neural Network 3-8-1', '-'*50)

    np.random.seed(17)  # początkowy wybór wag WAŻNE
                    # możliwe że potrzebny będzie inny seed
                    # rożne implementacje w różnych wersjach

    # X = np.array([[0, 0, 0],
    #               [0, 0, 1],
    #               [0, 1, 0],
    #               [0, 1, 1],
    #               [1, 0, 0],
    #               [1, 0, 1],
    #               [1, 1, 0],
    #               [1, 1, 1]])
    
    # y = np.array([[0], [0], [0], [1], [0], [1], [1], [0]])

    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork381(X, y)

    for _ in range(5000):
        nn.feedforward()
        nn.backprop()

    # Przed print'ami dodajmy line z ustaleniem precyzji
    np.set_printoptions(precision=3, suppress=True)
    print('weights1:\n', nn.weights1)
    print('weights2:\n', nn.weights2)

    print('output:\n', nn.output)
    print('solve([0, 1, 0]):\n', nn.solve(np.array([0, 1, 0])))

def test_3841():
    print('-'*50, 'Neural Network 3-8-4-1', '-'*50)
    np.random.seed(17)  # początkowy wybór wag WAŻNE
                    # możliwe że potrzebny będzie inny seed
                    # rożne implementacje w różnych wersjach

    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork3841(X, y)

    for _ in range(5000):
        nn.feedforward()
        nn.backprop()

    # Przed print'ami dodajmy line z ustaleniem precyzji
    np.set_printoptions(precision=3, suppress=True)
    print('weights1:\n', nn.weights1)
    print('weights2:\n', nn.weights2)
    print('weights3:\n', nn.weights3)

    print('output:\n', nn.output)


def test_custom_381():
    print('-'*50, 'Neural Network Custom (3-8-1)', '-'*50)

    np.random.seed(17)  # początkowy wybór wag WAŻNE
                    # możliwe że potrzebny będzie inny seed
                    # rożne implementacje w różnych wersjach

    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(X, y, [8])

    for _ in range(5000):
        nn.feedforward()
        nn.backprop()

    # Przed print'ami dodajmy line z ustaleniem precyzji
    np.set_printoptions(precision=3, suppress=True)
    for m, weight in enumerate(nn.weights):
        print(f'weight[{m}]:', weight, sep='\n')

    # print('nn.layer:\n', nn.layer)
    # print('layer_sizes:\n', nn.layer_sizes)
    print('output:\n', nn.output)


def test_custom_3841():
    print('-'*50, 'Neural Network Custom (3-8-4-1)', '-'*50)

    np.random.seed(17)  # początkowy wybór wag WAŻNE
                    # możliwe że potrzebny będzie inny seed
                    # rożne implementacje w różnych wersjach

    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(X, y, [8, 4])

    for _ in range(5000):
        nn.feedforward()
        nn.backprop()

    # Przed print'ami dodajmy line z ustaleniem precyzji
    np.set_printoptions(precision=3, suppress=True)
    for m, weight in enumerate(nn.weights):
        print(f'weight[{m}]:', weight, sep='\n')

    # print('nn.layer:\n', nn.layer)
    # print('layer_sizes:\n', nn.layer_sizes)
    print('output:\n', nn.output)
    


def main():
    test_381()
    test_custom_381()

    test_3841()
    test_custom_3841()

if __name__ == "__main__":
    main()
