import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import typing
import os
import networkx as nx


class Neuratron(nn.Module):
    def __init__(self, total_size: typing.Tuple[int, int]):
        super(Neuratron, self).__init__()
        
        # Inicializa o grande tensor de parâmetros
        self.total_weight = nn.Parameter(torch.Tensor(total_size[1], total_size[0]))
        self.total_bias = nn.Parameter(torch.Tensor(total_size[1]))
        
        # Lista de blocos livres: cada bloco é ((in_start, in_end), (out_start, out_end))
        self.free_blocks = [((0, total_size[0]), (0, total_size[1]))]
        
        # Dicionário de alocações
        self.allocations = {}
        
        # Inicialização dos pesos
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Inicializa os parâmetros usando Kaiming initialization"""
        # Inicialização do peso
        nn.init.kaiming_uniform_(self.total_weight, a=math.sqrt(5))
        
        # Inicialização do bias
        if self.total_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.total_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.total_bias, -bound, bound)

    def allocate(self, name: str, input_shape: int, output_shape: int) -> bool:
        """Aloca um bloco de parâmetros para uma nova camada"""
        
        # Encontrar o melhor bloco livre que caiba a alocação
        best_block_idx = -1
        best_fit_score = float('inf')
        
        for i, ((in_start, in_end), (out_start, out_end)) in enumerate(self.free_blocks):
            in_available = in_end - in_start
            out_available = out_end - out_start
            
            if in_available >= input_shape and out_available >= output_shape:
                # Calcula o "desperdício" (quanto sobra)
                waste = (in_available - input_shape) + (out_available - output_shape)
                if waste < best_fit_score:
                    best_fit_score = waste
                    best_block_idx = i
        
        if best_block_idx == -1:
            raise RuntimeError("Não há espaço suficiente para alocar")
        
        # Recupera o bloco escolhido
        (in_start, in_end), (out_start, out_end) = self.free_blocks.pop(best_block_idx)
        
        # Define os limites da alocação
        alloc_in_end = in_start + input_shape
        alloc_out_end = out_start + output_shape
        
        # Registra a alocação
        self.allocations[name] = (
            (in_start, alloc_in_end), 
            (out_start, alloc_out_end)
        )
        
        # 🔥 DIVISÃO DO ESPAÇO LIVRE 🔥
        # Criamos os 3 novos blocos livres resultantes da alocação
        
        # Bloco 1: Direita do alocado (mesma linha, coluna à direita)
        if alloc_in_end < in_end:
            free1 = ((alloc_in_end, in_end), (out_start, out_end))
            self.free_blocks.append(free1)
        
        # Bloco 2: Abaixo do alocado (mesma coluna, linha abaixo)  
        if alloc_out_end < out_end:
            free2 = ((in_start, in_end), (alloc_out_end, out_end))
            self.free_blocks.append(free2)
        
        # Bloco 3: Diagonal inferior direita (apenas se ambos sobrarem)
        if alloc_in_end < in_end and alloc_out_end < out_end:
            free3 = ((alloc_in_end, in_end), (alloc_out_end, out_end))
            self.free_blocks.append(free3)
        
        return True

    def get_layer(self, name: str):
        """Retorna os pesos e bias para uma camada alocada"""
        if name not in self.allocations:
            raise ValueError(f"Alocação {name} não encontrada")
        
        (in_start, in_end), (out_start, out_end) = self.allocations[name]
        
        weight = self.total_weight[out_start:out_end, in_start:in_end]
        bias = self.total_bias[out_start:out_end]
        
        return weight, bias

    def forward(self, x, allocation_name: str):
        """Executa a operação linear usando os parâmetros alocados"""
        weight, bias = self.get_layer(allocation_name)
        return F.linear(x, weight, bias)

    def print_memory_map(self):
        """Função auxiliar para visualizar o mapa de memória"""
        print("\n📊 Mapa de Memória do Neuratron:")
        print(f"Peso total: {self.total_weight.shape}")
        print(f"Alocações: {len(self.allocations)}")
        
        for name, ((in_s, in_e), (out_s, out_e)) in self.allocations.items():
            print(f"  {name}: entrada [{in_s}:{in_e}], saída [{out_s}:{out_e}]")
        
        print(f"Blocos livres: {len(self.free_blocks)}")
        for i, ((in_s, in_e), (out_s, out_e)) in enumerate(self.free_blocks):
            print(f"  Livre {i}: entrada [{in_s}:{in_e}], saída [{out_s}:{out_e}]")
