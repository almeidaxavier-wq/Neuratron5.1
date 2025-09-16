import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

import os
import random
import math
import typing
from collections import deque

from neuratron import Neuratron

class BrainSys(nn.Module):
    def __init__(self, neura: Neuratron, learning_rate: float, size: typing.Tuple, n_layers: int, optimizer: typing.Any):
        super().__init__()  # Adicionando herança de nn.Module
        self.neura = neura
        self.lr = learning_rate
        self.G = nx.DiGraph()
        self.history = deque()
        self.optimizer = optimizer
        self.n_layers = n_layers
        self.size = size

        # Alocar camadas
        for i in range(n_layers):
            layer_name = f"Neura-{i}"
            self.neura.allocate(layer_name, size[0], size[1])
            self.G.add_node(layer_name, prob=1.0)  # Inicializar probabilidade

        # Conectar camadas compatíveis
        layers = list(self.neura.allocations.items())
        for i, (name_i, pos_i) in enumerate(layers):
            for j, (name_j, pos_j) in enumerate(layers):
                if i != j:
                    # Verificar compatibilidade dimensional
                    output_size_i = pos_i[0][1] - pos_i[0][0]
                    input_size_j = pos_j[1][1] - pos_j[1][0]
                    
                    if output_size_i == input_size_j:
                        self.G.add_edge(name_i, name_j, weight=1.0)

    def forward(self, X: torch.Tensor, initial_alloc: str, depth: int):
        self.history = deque([initial_alloc])  # Resetar histórico
        current = initial_alloc
        result = X

        for i in range(depth):
            # Obter parâmetros da camada atual
            if current not in self.neura.allocations:
                raise ValueError(f"Alocação {current} não encontrada")
                
            idx = self.neura.allocations[current]
            w = self.neura.total_weight[idx[0][0]:idx[0][1], idx[1][0]:idx[1][1]]
            b = self.neura.total_bias[idx[0][0]:idx[0][1]]
            
            # Aplicar transformação linear
            result = F.linear(result, w, b)
            
            # Aplicar ativação (exceto na última camada)
            if i < depth - 1:
                result = F.relu(result)
            
            # Escolher próxima camada
            neighbors = list(self.G.neighbors(current))
            if not neighbors:
                break  # Sem mais camadas para seguir
                
            # Obter probabilidades dos vizinhos
            probs = [self.G[current][nbr]['weight'] for nbr in neighbors]
            
            # Normalizar probabilidades
            probs_sum = sum(probs)
            if probs_sum > 0:
                probs = [p / probs_sum for p in probs]
            else:
                probs = [1.0 / len(neighbors)] * len(neighbors)  # Distribuição uniforme
            
            # Escolher próxima camada
            next_alloc = random.choices(neighbors, weights=probs, k=1)[0]
            self.history.append(next_alloc)
            current = next_alloc

        return result

    def backpropagation(self, output: torch.Tensor, tgt: torch.Tensor):
        self.optimizer.zero_grad()
        
        # Calcular perda
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, tgt)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Atualizar probabilidades baseadas no desempenho
        # Camadas que contribuíram para bom resultado têm probabilidade aumentada
        with torch.no_grad():
            # Calcular magnitude média do gradiente como medida de contribuição
            grad_magnitudes = {}
            for layer_name in self.history:
                if layer_name in self.neura.allocations:
                    idx = self.neura.allocations[layer_name]
                    w_grad = self.neura.total_weight.grad[idx[0][0]:idx[0][1], idx[1][0]:idx[1][1]]
                    if w_grad is not None:
                        grad_magnitude = torch.mean(torch.abs(w_grad)).item()
                        grad_magnitudes[layer_name] = grad_magnitude
            
            # Normalizar magnitudes e atualizar probabilidades
            if grad_magnitudes:
                max_magnitude = max(grad_magnitudes.values())
                for layer_name, magnitude in grad_magnitudes.items():
                    # Camadas com gradientes maiores (mais influentes) têm probabilidade aumentada
                    update_factor = 1.0 + (magnitude / max_magnitude) * 0.1
                    self.G.nodes[layer_name]['prob'] *= update_factor
                    
                    # Atualizar também pesos das arestas
                    for pred in self.G.predecessors(layer_name):
                        self.G[pred][layer_name]['weight'] *= update_factor
        
        return loss.item()

    def get_path_probability(self, path):
        """Calcula a probabilidade total de um caminho"""
        total_prob = 1.0
        for i in range(len(path) - 1):
            source, target = path[i], path[i+1]
            if self.G.has_edge(source, target):
                total_prob *= self.G[source][target]['weight']
        return total_prob

    def reinforce_path(self, path, factor=1.1):
        """Reforça um caminho bem-sucedido"""
        for i in range(len(path) - 1):
            source, target = path[i], path[i+1]
            if self.G.has_edge(source, target):
                self.G[source][target]['weight'] *= factor
                self.G.nodes[target]['prob'] *= factor

    def penalize_path(self, path, factor=0.9):
        """Penaliza um caminho mal-sucedido"""
        for i in range(len(path) - 1):
            source, target = path[i], path[i+1]
            if self.G.has_edge(source, target):
                self.G[source][target]['weight'] *= factor
                self.G.nodes[target]['prob'] *= factor
                