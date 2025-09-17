from neuratron import Neuratron
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

from scipy.special import softmax

import unittest
import random
import numpy as np

# config.py - Configurações do experimento
EXPERIMENT_CONFIG = {
    'size':1000,
    'learning_rate':0.001,
    'n_layers': 10,
    'epochs': 100,
    'depth': 10

}

class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        # INICIALIZANDO MODELO
        self.model = Neuratron(EXPERIMENT_CONFIG['size'], EXPERIMENT_CONFIG['learning_rate'])      

        return super().setUp()

    def testTrainDigits(self):        
        # GERAR DATASET DIGITS
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40, train_size=0.2)

        # ALOCANDO MEMÓRIA
        for i in range(EXPERIMENT_CONFIG['n_layers']):
            inp = random.choice([20, 10, X_train.shape[1]])
            out = 10

            alloc = self.model.allocate(f'Neura-{i}', inp, out)
            if not alloc:
                break  

        self.model.allocate('Input', X_train.shape[1], 20)
        self.model.allocate('Output', 20, 10)

        print(X_train.shape, y_train.shape)

        def train(model:Neuratron, epochs:int=EXPERIMENT_CONFIG['epochs'], depth:int=EXPERIMENT_CONFIG['depth']):
            for _ in range(epochs):
                res = model.forward(X_train, 'Input', depth=depth)
                res = softmax(res)
                loss = log_loss(y_train, res)

                #print([np.max(row) for row in res])
                
                print("#" + '='*20 + '#')
                print("ACCURACY: ", accuracy_score(y_train, [np.max(row).astype(int) for row in res]))
                print("LOSS: ", loss)

                model.backpropagation(loss=loss)

        def test(model:Neuratron, depth:int=EXPERIMENT_CONFIG['depth']):
            res = model.forward(X_test, 'Input', depth=depth)
            loss = log_loss(y_train, res)

            print("#" + '='*20 + '#')
            print("ACCURACY: ", accuracy_score(y_test, [np.max(row) for row in res]))
            print("LOSS: ", loss)

        train(self.model)
        test(self.model)
