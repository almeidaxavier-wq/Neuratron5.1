import torch
import torch.nn as nn
import torch.nn.functional as F
from neura.neuratron import Neuratron  # Importe sua classe aqui

def test_initialization():
    """Testa se o Neuratron inicializa corretamente"""
    print("=== Teste de Inicialização ===")
    
    total_size = (100, 50)
    model = Neuratron(total_size)
    
    # Verifica se os parâmetros foram criados
    assert model.total_weight.shape == (50, 100), f"Peso total shape errado: {model.total_weight.shape}"
    assert model.total_bias.shape == (50,), f"Bias total shape errado: {model.total_bias.shape}"
    
    # Verifica se há 1 bloco livre inicial
    assert len(model.free_blocks) == 1, f"Blocos livres iniciais: {len(model.free_blocks)}"
    assert model.free_blocks[0] == ((0, 100), (0, 50)), f"Bloco livre inicial errado: {model.free_blocks[0]}"
    
    print("✅ Inicialização OK")

def test_basic_allocation():
    """Testa uma alocação básica"""
    print("\n=== Teste de Alocação Básica ===")
    
    model = Neuratron((100, 50))
    
    # Aloca uma camada 20x10
    success = model.allocate("camada1", 20, 10)
    assert success, "Falha na alocação"
    
    # Verifica se a alocação foi registrada
    assert "camada1" in model.allocations, "Alocação não registrada"
    
    # Verifica os índices da alocação
    alloc = model.allocations["camada1"]
    assert alloc == ((0, 20), (0, 10)), f"Alocação errada: {alloc}"
    
    # Verifica se criou blocos livres
    assert len(model.free_blocks) >= 1, f"Deve ter pelo menos 1 bloco livre, got {len(model.free_blocks)}"
    
    print("✅ Alocação básica OK")
    model.print_memory_map()

def test_forward_pass():
    """Testa se o forward pass funciona"""
    print("\n=== Teste de Forward Pass ===")
    
    model = Neuratron((10, 5))
    model.allocate("test_layer", 3, 2)
    
    # Pega os pesos manualmente para verificação
    weight, bias = model.get_layer("test_layer")
    assert weight.shape == (2, 3), f"Peso shape errado: {weight.shape}"
    assert bias.shape == (2,), f"Bias shape errado: {bias.shape}"
    
    # Testa o forward
    x = torch.randn(4, 3)  # batch_size=4, input_size=3
    output = model.forward(x, "test_layer")
    
    assert output.shape == (4, 2), f"Output shape errado: {output.shape}"
    
    # Verifica manualmente a operação linear
    expected_output = x @ weight.T + bias
    torch.testing.assert_close(output, expected_output)
    
    print("✅ Forward pass OK")

def test_training():
    """Testa se o treinamento funciona"""
    print("\n=== Teste de Treinamento ===")
    
    model = Neuratron((10, 8))
    model.allocate("train_layer", 5, 3)
    
    # Configura otimizador
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Dados de exemplo
    x = torch.randn(8, 5)
    y_target = torch.randn(8, 3)
    
    # Loop de treino simples
    initial_loss = None
    for epoch in range(5):
        optimizer.zero_grad()
        output = model.forward(x, "train_layer")
        loss = F.mse_loss(output, y_target)
        loss.backward()
        optimizer.step()
        
        if initial_loss is None:
            initial_loss = loss.item()
        
        print(f"Época {epoch}, Loss: {loss.item():.4f}")
    
    # Verifica se o loss diminuiu
    final_loss = loss.item()
    assert final_loss < initial_loss, f"Loss não diminuiu: {initial_loss} -> {final_loss}"
    
    print("✅ Treinamento OK")

# Executa os testes
if __name__ == "__main__":
    print("🧪 Testando Neuratron...")
    
    try:
        test_initialization()
        test_basic_allocation()
        test_forward_pass()
        test_training()
        print("\n🎉 Todos os testes passaram!")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()