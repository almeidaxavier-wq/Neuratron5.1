import numpy as np
import networkx as nx
import math
import random
import typing
import os


class Neuratron:
    def  __init__(self, size:int, lr:float):
        # Matriz Omega
        tot_mtrx = np.random.randn(size, size)

        # Alocacoes e espaços livres
        self.allocations = {}
        self.free = [tot_mtrx]

        # Learning rate
        self.lr = lr

        # Grafo de conexões cerebrais
        self.G = nx.DiGraph()
    
    def allocate(self, allocation_name:str, input_size:int, output_size:int) -> bool:
        # Itera sobre as partições livres
        for i, f in enumerate(self.free):
            rows, cols = f.shape

            if input_size <= rows and output_size <= cols:
                # Aloca a matriz

                self.free.pop(i)
                self.allocations[allocation_name] = {
                    'weight': f[:input_size, :output_size],
                    'bias': np.random.randn(output_size),
                    'prob': 1,
                    'usages': 0                
                }

                self.free.extend([f[:input_size, output_size:], f[input_size:, output_size:], f[input_size:, :output_size]])
                for alloc, params in self.allocations.items():

                    _ , shape_in_col = self.allocations[allocation_name]['weight'].shape
                    shape_out_row, _ = params['weight'].shape

                    if shape_in_col == shape_out_row and alloc != allocation_name:
                        self.G.add_edge(allocation_name, alloc)
                
                return True
        return False
    
    def forward(self, X:np.ndarray, first_alloc:str, depth:int=10) -> np.ndarray:
        res = np.copy(X)
        w = self.allocations[first_alloc]['weight']
        b = self.allocations[first_alloc]['bias']
        
        self.history = [first_alloc]
        self.allocations[first_alloc]['usages'] += 1

        for _ in range(depth):
            res = res @ w + b 
            weights = [self.allocations[alloc]['prob'] for alloc in self.G.neighbors(first_alloc)]
            if len(weights) == 0:
                break

            chosed = random.choices(list(self.G.neighbors(first_alloc)), weights=weights, k=1)[0]

            self.allocations[chosed]['usages'] += 1

            w = self.allocations[chosed]['weight']
            b = self.allocations[chosed]['bias']

            self.history.append(chosed)
            first_alloc = chosed

        return res

    # Algoritmo Backpropagation
    def backpropagation(self, loss:np.ndarray) -> None:
        grad = np.gradient(loss)  

        for hist in reversed(self.history):
            print(grad.shape)
            self.allocations[hist]['weight'] -= self.lr * (grad @ self.allocations[hist]['weight']).astype(float)
            self.allocations[hist]['bias'] = grad.astype(float)
            self.allocations[hist]['prob'] += np.sum(grad)**-1
            
            grad = grad @ self.allocations[hist]['weight'].T   
