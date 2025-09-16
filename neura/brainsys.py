import torch
import torch.nn as nn
import torch.optim as optim
import typing
import random
import networkx as nx
import matplotlib.pyplot as plt

# Classe Neuratron (assumindo que está no mesmo arquivo ou importada)
from neuratron import Neuratron

class BrainSys(nn.Module):
    def __init__(self, neura: Neuratron, optimizer: typing.Any):
        super().__init__()        
        self.layers = neura
        self.n_layers = 0
        self.optim = optimizer
        
        # Usamos um grafo interno para gerenciar conexões
        self.graph = nx.DiGraph()
        self.allocation_names = []  # Lista para manter a ordem das alocações
        
        # Histórico de recompensas para tracking
        self.reward_history = []
        self.connection_history = []

    def allocate(self, input_shape: int, output_shape: int) -> str:
        """Aloca uma nova camada e retorna seu nome"""
        layer_name = f'Neura-{self.n_layers}'
        self.layers.allocate(layer_name, input_shape, output_shape)
        self.graph.add_node(layer_name)
        self.allocation_names.append(layer_name)
        self.n_layers += 1
        return layer_name

    def connect_layers(self, source_layer: str, target_layer: str, prob: float = 0.5):
        """Conecta duas camadas com uma probabilidade"""
        if source_layer not in self.layers.allocations or target_layer not in self.layers.allocations:
            raise ValueError("Camadas não alocadas")
        
        # Verifica compatibilidade dimensional
        source_out_size = self.layers.allocations[source_layer][1][1] - self.layers.allocations[source_layer][1][0]
        target_in_size = self.layers.allocations[target_layer][0][1] - self.layers.allocations[target_layer][0][0]
        
        if source_out_size != target_in_size:
            raise ValueError(f"Incompatibilidade dimensional: {source_out_size} != {target_in_size}")
        
        self.graph.add_edge(source_layer, target_layer, prob=prob, weight=1.0)

    def generate_connections(self, connection_prob: float = 0.3):
        """Gera conexões automaticamente entre camadas compatíveis"""
        for i, source_name in enumerate(self.allocation_names):
            source_alloc = self.layers.allocations[source_name]
            source_out_size = source_alloc[1][1] - source_alloc[1][0]
            
            for j, target_name in enumerate(self.allocation_names):
                if i == j:
                    continue  # Não conectar consigo mesmo
                
                target_alloc = self.layers.allocations[target_name]
                target_in_size = target_alloc[0][1] - target_alloc[0][0]
                
                if source_out_size == target_in_size and random.random() < connection_prob:
                    prob = random.uniform(0.1, 0.9)
                    self.connect_layers(source_name, target_name, prob)

    def forward(self, initial_alloc: str, depth: int, X: torch.Tensor):
        """Propaga a entrada através do grafo por 'depth' passos"""
        if initial_alloc not in self.layers.allocations:
            raise ValueError(f"Alocação inicial {initial_alloc} não existe")
        
        current_alloc = initial_alloc
        result = X
        path_taken = [current_alloc]  # Rastreia o caminho percorrido
        
        for d in range(depth):
            # Executa a camada atual
            result = self.layers.forward(result, current_alloc)
            
            # Aplica não-linearidade
            result = torch.relu(result)
            
            # Decide próxima camada (se houver conexões)
            neighbors = list(self.graph.successors(current_alloc))
            if not neighbors:
                break  # Sem mais conexões
                
            # Escolhe a próxima camada baseado nas probabilidades das arestas
            edge_probs = []
            for neighbor in neighbors:
                prob = self.graph[current_alloc][neighbor]['prob']
                edge_probs.append(prob)
            
            # Amostra a próxima camada
            next_alloc = random.choices(neighbors, weights=edge_probs, k=1)[0]
            current_alloc = next_alloc
            path_taken.append(current_alloc)
        
        return result, path_taken

    def update_connection_strengths(self, reward: float, path_taken: list, 
                                  decay: float = 0.95, learning_rate: float = 0.1):
        """
        Atualiza as probabilidades das conexões com base na recompensa
        e no caminho percorrido.
        
        Args:
            reward: Recompensa do episódio (0-1)
            path_taken: Lista de camadas visitadas durante o forward
            decay: Fator de decaimento das probabilidades antigas
            learning_rate: Taxa de aprendizado para novas experiências
        """
        # Reforça todas as conexões no caminho percorrido
        for i in range(len(path_taken) - 1):
            current_layer = path_taken[i]
            next_layer = path_taken[i + 1]
            
            if self.graph.has_edge(current_layer, next_layer):
                current_prob = self.graph[current_layer][next_layer]['prob']
                
                # Atualização: combina experiência anterior com nova recompensa
                updated_prob = current_prob * decay + reward * learning_rate * (1 - decay)
                
                # Mantém dentro de limites razoáveis
                updated_prob = max(0.1, min(0.9, updated_prob))
                
                self.graph[current_layer][next_layer]['prob'] = updated_prob
        
        # Registra para análise
        self.reward_history.append(reward)
        self.connection_history.append(self.get_connection_stats())

    def get_connection_stats(self):
        """Retorna estatísticas das conexões para monitoramento"""
        probs = []
        for u, v, data in self.graph.edges(data=True):
            probs.append(data['prob'])
        
        return {
            'mean_prob': sum(probs) / len(probs) if probs else 0,
            'min_prob': min(probs) if probs else 0,
            'max_prob': max(probs) if probs else 0,
            'total_connections': len(probs)
        }

    def backprop(self, loss: torch.Tensor):
        """Backpropagation padrão"""
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def visualize_learning(self):
        """Visualiza o progresso do aprendizado das conexões"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Gráfico de recompensas
        ax1.plot(self.reward_history)
        ax1.set_title('Histórico de Recompensas')
        ax1.set_xlabel('Episódio')
        ax1.set_ylabel('Recompensa')
        ax1.grid(True)
        
        # Gráfico de estatísticas das conexões
        episodes = range(len(self.connection_history))
        mean_probs = [stats['mean_prob'] for stats in self.connection_history]
        min_probs = [stats['min_prob'] for stats in self.connection_history]
        max_probs = [stats['max_prob'] for stats in self.connection_history]
        
        ax2.plot(episodes, mean_probs, label='Média', linewidth=2)
        ax2.fill_between(episodes, min_probs, max_probs, alpha=0.3, label='Variação')
        ax2.set_title('Evolução das Probabilidades de Conexão')
        ax2.set_xlabel('Episódio')
        ax2.set_ylabel('Probabilidade')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def print_connection_stats(self):
        """Imprime estatísticas atuais das conexões"""
        stats = self.get_connection_stats()
        print(f"\n📊 Estatísticas de Conexões:")
        print(f"Total de conexões: {stats['total_connections']}")
        print(f"Probabilidade média: {stats['mean_prob']:.3f}")
        print(f"Probabilidade mínima: {stats['min_prob']:.3f}")
        print(f"Probabilidade máxima: {stats['max_prob']:.3f}")
        
        # Mostra as 5 conexões mais fortes
        print("\n🔝 Top 5 conexões mais fortes:")
        edges_with_probs = [(u, v, data['prob']) for u, v, data in self.graph.edges(data=True)]
        edges_with_probs.sort(key=lambda x: x[2], reverse=True)
        
        for u, v, prob in edges_with_probs[:5]:
            print(f"  {u} → {v}: {prob:.3f}")