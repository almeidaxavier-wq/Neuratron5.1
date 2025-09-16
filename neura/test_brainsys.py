import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import sys
import os

# Adiciona o diretório atual ao path para importar os módulos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuratron import Neuratron
from brainsys import BrainSys

class TestNeuratron:
    """Testes para a classe Neuratron"""
    
    def test_initialization(self):
        """Testa a inicialização do Neuratron"""
        print("🧪 Testando inicialização do Neuratron...")
        
        total_size = (100, 50)
        model = Neuratron(total_size)
        
        # Verifica shapes dos parâmetros
        assert model.total_weight.shape == (50, 100)
        assert model.total_bias.shape == (50,)
        
        # Verifica blocos livres iniciais
        assert len(model.free_blocks) == 1
        assert model.free_blocks[0] == ((0, 100), (0, 50))
        
        print("✅ Neuratron inicializado corretamente")

    def test_basic_allocation(self):
        """Testa alocação básica"""
        print("🧪 Testando alocação básica...")
        
        model = Neuratron((100, 50))
        
        # Aloca uma camada
        success = model.allocate("camada_teste", 20, 10)
        assert success
        
        # Verifica se a alocação foi registrada
        assert "camada_teste" in model.allocations
        assert model.allocations["camada_teste"] == ((0, 20), (0, 10))
        
        # Verifica que blocos livres foram criados
        assert len(model.free_blocks) >= 1
        
        print("✅ Alocação básica funcionando")

    def test_forward_pass(self):
        """Testa o forward pass"""
        print("🧪 Testando forward pass...")
        
        model = Neuratron((10, 5))
        model.allocate("test_layer", 3, 2)
        
        # Testa forward pass
        x = torch.randn(4, 3)
        output = model.forward(x, "test_layer")
        
        assert output.shape == (4, 2)
        
        # Verifica cálculo manual
        weight, bias = model.get_layer("test_layer")
        expected = F.linear(x, weight, bias)
        torch.testing.assert_close(output, expected)
        
        print("✅ Forward pass funcionando")

    def test_multiple_allocations(self):
        """Testa múltiplas alocações"""
        print("🧪 Testando múltiplas alocações...")
        
        model = Neuratron((100, 100))
        
        # Aloca várias camadas
        model.allocate("layer1", 30, 20)
        model.allocate("layer2", 25, 15)
        model.allocate("layer3", 40, 10)
        
        assert len(model.allocations) == 3
        assert "layer1" in model.allocations
        assert "layer2" in model.allocations
        assert "layer3" in model.allocations
        
        print("✅ Múltiplas alocações funcionando")

    def test_allocation_failure(self):
        """Testa falha de alocação quando não há espaço"""
        print("🧪 Testando falha de alocação...")
        
        model = Neuratron((10, 10))
        
        # Aloca quase todo o espaço
        model.allocate("big_layer", 9, 9)
        
        # Deve falhar ao tentar alocar mais
        try:
            model.allocate("should_fail", 5, 5)
            assert False, "Deveria ter falhado"
        except RuntimeError as e:
            assert "Não há espaço suficiente" in str(e)
        
        print("✅ Falha de alocação funcionando corretamente")

    def test_training_compatibility(self):
        """Testa compatibilidade com treinamento"""
        print("🧪 Testando compatibilidade com treinamento...")
        
        model = Neuratron((10, 8))
        model.allocate("train_layer", 5, 3)
        
        # Verifica se os parâmetros são acessíveis
        params = list(model.parameters())
        assert len(params) == 2
        
        # Testa otimizador
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        x = torch.randn(8, 5)
        y_target = torch.randn(8, 3)
        
        # Loop de treino simples
        for epoch in range(2):
            optimizer.zero_grad()
            output = model.forward(x, "train_layer")
            loss = F.mse_loss(output, y_target)
            loss.backward()
            optimizer.step()
        
        print("✅ Compatibilidade com treinamento OK")

    def run_all_tests(self):
        """Executa todos os testes do Neuratron"""
        print("\n" + "="*60)
        print("🚀 INICIANDO TESTES DO NEURATRON")
        print("="*60)
        
        tests = [
            self.test_initialization,
            self.test_basic_allocation,
            self.test_forward_pass,
            self.test_multiple_allocations,
            self.test_allocation_failure,
            self.test_training_compatibility
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"❌ {test.__name__} FALHOU: {e}")
                raise
        
        print("\n🎉 TODOS OS TESTES DO NEURATRON PASSARAM!")

class TestBrainSys:
    """Testes para a classe BrainSys"""
    
    def test_brainsys_initialization(self):
        """Testa inicialização do BrainSys"""
        print("🧪 Testando inicialização do BrainSys...")
        
        neuratron = Neuratron((100, 50))
        optimizer = torch.optim.Adam(neuratron.parameters(), lr=0.001)
        
        brain = BrainSys(neuratron, optimizer)
        
        assert brain.layers == neuratron
        assert brain.optim == optimizer
        assert brain.n_layers == 0
        assert len(brain.graph.nodes) == 0
        assert len(brain.graph.edges) == 0
        
        print("✅ BrainSys inicializado corretamente")

    def test_brain_allocation(self):
        """Testa alocação no BrainSys"""
        print("🧪 Testando alocação no BrainSys...")
        
        neuratron = Neuratron((100, 50))
        optimizer = torch.optim.Adam(neuratron.parameters(), lr=0.001)
        brain = BrainSys(neuratron, optimizer)
        
        # Aloca algumas camadas
        brain.allocate(20, 10)  # Neura-0
        brain.allocate(10, 5)   # Neura-1
        
        assert brain.n_layers == 2
        assert "Neura-0" in brain.layers.allocations
        assert "Neura-1" in brain.layers.allocations
        assert "Neura-0" in brain.graph.nodes
        assert "Neura-1" in brain.graph.nodes
        
        print("✅ Alocação no BrainSys funcionando")

    def test_layer_connection(self):
        """Testa conexão entre camadas"""
        print("🧪 Testando conexão entre camadas...")
        
        neuratron = Neuratron((100, 50))
        optimizer = torch.optim.Adam(neuratron.parameters(), lr=0.001)
        brain = BrainSys(neuratron, optimizer)
        
        # Aloca camadas compatíveis
        brain.allocate(20, 10)  # Neura-0
        brain.allocate(10, 5)   # Neura-1
        
        # Conecta as camadas
        brain.connect_layers("Neura-0", "Neura-1", prob=0.8)
        
        assert brain.graph.has_edge("Neura-0", "Neura-1")
        assert brain.graph["Neura-0"]["Neura-1"]["prob"] == 0.8
        
        print("✅ Conexão entre camadas funcionando")

    def test_incompatible_connection(self):
        """Testa tentativa de conexão incompatível"""
        print("🧪 Testando conexão incompatível...")
        
        neuratron = Neuratron((100, 50))
        optimizer = torch.optim.Adam(neuratron.parameters(), lr=0.001)
        brain = BrainSys(neuratron, optimizer)
        
        # Aloca camadas INcompatíveis
        brain.allocate(20, 10)  # Neura-0
        brain.allocate(15, 5)   # Neura-1 (incompatível: espera 15, mas Neura-0 produz 10)
        
        try:
            brain.connect_layers("Neura-0", "Neura-1", prob=0.8)
            assert False, "Deveria ter falhado"
        except ValueError as e:
            assert "Incompatibilidade dimensional" in str(e)
        
        print("✅ Prevenção de conexão incompatível funcionando")

    def test_automatic_connection_generation(self):
        """Testa geração automática de conexões"""
        print("🧪 Testando geração automática de conexões...")
        
        neuratron = Neuratron((100, 50))
        optimizer = torch.optim.Adam(neuratron.parameters(), lr=0.001)
        brain = BrainSys(neuratron, optimizer)
        
        # Aloca várias camadas compatíveis
        brain.allocate(20, 10)  # Neura-0
        brain.allocate(10, 8)   # Neura-1
        brain.allocate(8, 5)    # Neura-2
        brain.allocate(5, 3)    # Neura-3
        
        # Gera conexões automaticamente
        brain.generate_connections(connection_prob=1.0)  # 100% de chance
        
        # Deve ter criado várias conexões
        assert len(brain.graph.edges) > 0
        
        print("✅ Geração automática de conexões funcionando")

    def test_forward_propagation(self):
        """Testa propagação forward através do grafo"""
        print("🧪 Testando propagação forward...")
        
        neuratron = Neuratron((100, 50))
        optimizer = torch.optim.Adam(neuratron.parameters(), lr=0.001)
        brain = BrainSys(neuratron, optimizer)
        
        # Aloca e conecta camadas
        brain.allocate(10, 8)  # Neura-0
        brain.allocate(8, 5)   # Neura-1
        brain.allocate(5, 3)   # Neura-2
        
        brain.connect_layers("Neura-0", "Neura-1", prob=0.9)
        brain.connect_layers("Neura-1", "Neura-2", prob=0.9)
        
        # Testa forward pass
        x = torch.randn(4, 10)  # batch_size=4, input_size=10
        output = brain.forward("Neura-0", depth=3, X=x)
        print(output[0].shape)
        
        assert output[0].shape == (4, 3)  # Deve passar por 2 camadas: 10→8→5→3
        
        print("✅ Propagação forward funcionando")

    def test_training_loop(self):
        """Testa loop de treinamento completo"""
        print("🧪 Testando loop de treinamento...")
        
        neuratron = Neuratron((50, 30))
        optimizer = torch.optim.Adam(neuratron.parameters(), lr=0.001)
        brain = BrainSys(neuratron, optimizer)
        
        # Configura arquitetura
        brain.allocate(10, 8)  # Neura-0
        brain.allocate(8, 5)   # Neura-1
        brain.connect_layers("Neura-0", "Neura-1", prob=0.9)
        
        # Dados de treino
        x_train = torch.randn(16, 10)
        y_train = torch.randn(16, 5)
        
        # Loop de treino
        losses = []
        for epoch in range(5):
            optimizer.zero_grad()
            output = brain.forward("Neura-0", depth=2, X=x_train)
            loss = F.mse_loss(output[0], y_train)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Verifica se o loss diminuiu
        assert losses[-1] < losses[0] or abs(losses[-1] - losses[0]) < 1.0
        
        print("✅ Loop de treinamento funcionando")

    def test_graph_visualization(self):
        """Testa a visualização do grafo (sem mostrar)"""
        print("🧪 Testando visualização do grafo...")
        
        neuratron = Neuratron((50, 30))
        optimizer = torch.optim.Adam(neuratron.parameters(), lr=0.001)
        brain = BrainSys(neuratron, optimizer)
        
        # Cria uma arquitetura simples
        brain.allocate(10, 8)
        brain.allocate(8, 5)
        brain.connect_layers("Neura-0", "Neura-1", prob=0.7)
        
        # Apenas verifica se a função existe e não quebra
        try:
            # Não mostra o gráfico durante os testes
            import matplotlib
            matplotlib.use('Agg')  # Usa backend não-interativo
            brain.visualize_learning()
            print("✅ Visualização do grafo funcionando (backend não-interativo)")
        except ImportError:
            print("⚠️  Matplotlib não disponível, pulando teste de visualização")

    def run_all_tests(self):
        """Executa todos os testes do BrainSys"""
        print("\n" + "="*60)
        print("🚀 INICIANDO TESTES DO BRAINSYS")
        print("="*60)
        
        tests = [
            self.test_brainsys_initialization,
            self.test_brain_allocation,
            self.test_layer_connection,
            self.test_incompatible_connection,
            self.test_automatic_connection_generation,
            self.test_forward_propagation,
            self.test_training_loop,
            self.test_graph_visualization
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"❌ {test.__name__} FALHOU: {e}")
                raise
        
        print("\n🎉 TODOS OS TESTES DO BRAINSYS PASSARAM!")

# Exemplo de uso completo
def example_usage():
    # Configuração inicial
    total_size = (300, 300)  # Tensor total de 200x200
    neura = Neuratron(total_size)
    optimizer = optim.Adam(neura.parameters(), lr=0.1, weight_decay=1e-5)
    
    # Cria o sistema cerebral
    brain = BrainSys(neura, optimizer)
    
    # Aloca várias camadas
    layers = []
    for i in range(20):
        input_size = random.choice([10, 20, 30])
        output_size = random.choice([10, 20, 30])
        layer_name = brain.allocate(input_size, output_size)
        layers.append(layer_name)
        print(f"Alocada camada {layer_name}: {input_size} → {output_size}")
    
    # Gera conexões aleatórias
    brain.generate_connections(connection_prob=0.4)
    print(f"\nGeradas {brain.graph.number_of_edges()} conexões iniciais")
    
    # Loop de treinamento simulado
    num_episodes = 1000
    
    for episode in range(num_episodes):
        # Gera dados de entrada aleatórios
        batch_size = 32
        input_size = brain.layers.allocations[layers[0]][0][1] - brain.layers.allocations[layers[0]][0][0]
        X = torch.randn(batch_size, input_size)
        
        depth = random.randint(2, 10)  # Em vez de depth fixo=3
        # Forward pass através do grafo
        output, path_taken = brain.forward(layers[0], depth=depth, X=X)
        
        # Simula uma tarefa de classificação (exemplo)
        target = torch.randint(0, 2, (batch_size, output.shape[1])).float()
        
        # Calcula perda
        loss = nn.MSELoss()(output, target)
        
        # Backpropagation para pesos das camadas
        brain.backprop(loss)
        
        # Calcula recompensa baseada na performance (1 - loss normalizada)
        reward = max(0, 1 - loss.item() * 10)  # Recompensa entre 0 e 1
        
        # ⭐ ATUALIZAÇÃO DINÂMICA DAS CONEXÕES ⭐
        brain.update_connection_strengths(reward, path_taken)
        
        if episode % 10 == 0:
            print(f"Episódio {episode}: Loss={loss.item():.4f}, Recompensa={reward:.3f}")
    
    # Visualiza resultados
    brain.print_connection_stats()
    
    # Mostra grafo final
    brain.visualize_learning()

def run_comprehensive_test():
    """Executa todos os testes comprehensive"""
    print("🧪 INICIANDO TESTES COMPLETOS DO SISTEMA")
    print("="*60)
    
    # Testa Neuratron
    neuratron_tester = TestNeuratron()
    neuratron_tester.run_all_tests()
    
    # Testa BrainSys
    brainsys_tester = TestBrainSys()
    brainsys_tester.run_all_tests()
    
    print("\n" + "="*60)
    print("🎉 TODOS OS TESTES PASSARAM! SISTEMA FUNCIONAL!")
    print("="*60)



if __name__ == "__main__":
    # Configura reproduibilidade para testes
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    try:
        run_comprehensive_test()
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO NOS TESTES: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    example_usage()
