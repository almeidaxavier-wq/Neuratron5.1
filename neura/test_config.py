import unittest
from neuratron import Neuratron, Neuron
import random


# config.py - Configurações do experimento
EXPERIMENT_CONFIG = {
    'n_neurons':1000,
    'learning_rate':0.1,
    'max_conns_per_neuron': 10,
    'epochs': 2000,
    'depth': 10

}

class TestBrain(unittest.TestCase):
    def setUp(self):
        self.neuratron = Neuratron(
            n_conns_initial=EXPERIMENT_CONFIG['n_neurons'],
            max_conns_per_neuron=EXPERIMENT_CONFIG['max_conns_per_neuron'],
            learning_rate=EXPERIMENT_CONFIG['learning_rate']
        )
        
        return super().setUp()

    def testTargeting(self):
        target = 2
        self.neuratron.inp = random.randint(0,3)

        for _ in range(EXPERIMENT_CONFIG['epochs']):
            self.neuratron.receive_input(target)

        n = random.choice(self.neuratron.chosing_neurons)
        neighbor = random.choice(list(self.neuratron.conns.neighbors(n)))
        n.process(1, self.neuratron.conns[n][neighbor]['weight'])
        self.assertAlmostEqual(n.x, target)

