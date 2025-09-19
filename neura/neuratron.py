import numpy as np
import networkx as nx
import threading
import random
import pickle
import time

from functools import partial
from scipy.special import expit
from scipy.special import softmax

def relu(inp: np.ndarray) -> np.ndarray:
    return np.array([i if i > 0 else 0 for i in inp])

def linear(inp:np.ndarray) -> np.ndarray:
    return inp

ACTIVATIONS = {
    'softmax': softmax,
    'sigmoid': expit,
    'linear': linear

}

class Neuratron:
    def __init__(self, n_conns_initial:int, max_conns_per_neuron:int, learning_rate:float):
        self.conns = nx.Graph()
        self.neurons = []
        self.learning_rate = learning_rate
        self.inp = 0
        self.wake_up = False

        # INICIALIZAÇÃO DOS NEURÔNIOS
        for _ in range(n_conns_initial):
            n = Neuron()
            self.conns.add_node(n)
            self.neurons.append(n)

        self.chosing_neurons = self.neurons.copy()

        # CRIAÇÃO DAS SINAPSES
        for N in self.conns.nodes():
            chosed_neurons = random.sample(list(self.conns.nodes()), k=max_conns_per_neuron)
            
            for chosed in chosed_neurons:
                self.conns.add_edge(N, chosed, weight=random.uniform(0,1))

    def receive_input(self, target:float) -> None:
        neuron = random.choice(self.chosing_neurons)
        neuron.x = self.inp if not self.wake_up else neuron.x
        self.wake_up = True
        neurons_processing = []
        threads = neurons_processing.copy()

        # PARA CADA VIZINHO, PASSAR O VALOR PARA PROCESSAMENTO
        for neighbor in self.conns.neighbors(neuron):
            thread = threading.Thread(target=partial(neighbor.process, neuron.x, self.conns[neuron][neighbor]['weight']))
            thread.start()            
            threads.append(thread)
            #print(type(thread))
            neurons_processing.append(neighbor)
        
        # TERMINAR THREADS
        for thread in threads:
            #print('A', type(thread))
            thread.join()

        for neur in neurons_processing:
            d_E = (target - neur.x) * (neur.x - self.inp)
            anterior_weight = self.conns[neuron][neur]['weight']
            self.conns[neuron][neur]['weight'] += self.learning_rate * d_E         
            post_weight = self.conns[neuron][neur]['weight']
            neur.bias += self.learning_rate * d_E

            print('CHANGE', anterior_weight, post_weight)
            #time.sleep(1)

        self.chosing_neurons = list(filter(lambda n:n.x > 0, [neighbor for neighbor in self.conns.neighbors(neuron)]))

class Neuron:
    def __init__(self) -> None:
        #self.lock = threading.Lock()
        self.bias = random.uniform(0, 1)
        self.is_calling = False
        self.x = 1

    def process(self, x:float, weight:float) -> float:
        self.is_calling = True
        #print(x, weight)
        self.x = ACTIVATIONS['sigmoid'](x * weight) + self.bias
