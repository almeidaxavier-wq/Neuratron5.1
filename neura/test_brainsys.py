import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from neuratron import Neuratron
from brainsys import BrainSys

class IrisClassificationExperiment:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Carregar e preparar dados
        self.load_data()
        
        # Configurar modelo
        self.setup_model()
        
    def load_data(self):
        """Carrega e prepara o dataset Iris"""
        print("Carregando dataset Iris...")
        
        # Carregar dados
        X, y = load_iris(return_X_y=True)
        
        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizar dados
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Converter para tensores
        self.X_train = torch.FloatTensor(X_train).to(self.device)
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_train = torch.LongTensor(y_train).to(self.device)
        self.y_test = torch.LongTensor(y_test).to(self.device)
        
        print(f"Dados de treino: {self.X_train.shape}")
        print(f"Dados de teste: {self.X_test.shape}")
        print(f"Classes: {np.unique(y)}")
        
    def setup_model(self):
        """Configura o modelo BrainSys"""
        print("Configurando modelo BrainSys...")
        
        # Configurar Neuratron
        total_size = 100
        self.neura = Neuratron(total_size=(total_size, total_size))
        
        # Configurar otimizador
        self.optimizer = optim.Adam([
            {'params': self.neura.total_weight, 'lr': 0.01},
            {'params': self.neura.total_bias, 'lr': 0.01}
        ])
        
        # Configurar BrainSys
        input_size = 4  # 4 features do Iris
        hidden_size = 10
        output_size = 3  # 3 classes
        n_layers = 6     # Número de camadas alocadas

        # Alocar camada de entrada e saída especiais
        self.neura.allocate("input_layer", hidden_size, input_size)
        self.neura.allocate("output_layer", output_size, hidden_size)

        self.model = BrainSys(
            neura=self.neura,
            learning_rate=0.01,
            size=(hidden_size, hidden_size),
            n_layers=n_layers,
            optimizer=self.optimizer
        ).to(self.device)
    
        
        # Adicionar ao grafo
        self.model.G.add_node("input_layer", prob=1.0)
        self.model.G.add_node("output_layer", prob=1.0)
        
        # Conectar camadas
        for layer in self.model.G.nodes():
            if layer != "output_layer":
                self.model.G.add_edge(layer, "output_layer", weight=1.0)
        
    def train(self, epochs=100, depth=10):
        """Treina o modelo"""
        print(f"Iniciando treinamento por {epochs} épocas...")
        
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # Modo de treino
            self.model.train()
            
            # Forward pass começando da camada de entrada
            outputs = self.model.forward(self.X_train, "input_layer", depth=depth)
            
            # Calcular perda
            loss = nn.CrossEntropyLoss()(outputs, self.y_train)
            
            # Backward pass
            self.model.backpropagation(outputs, self.y_train)
            
            # Calcular acurácia
            _, preds = torch.max(outputs, 1)
            accuracy = (preds == self.y_train).float().mean()
            
            # Avaliar no teste periodicamente
            if epoch % 10 == 0:
                test_acc = self.evaluate()
                test_accuracies.append(test_acc)
                
                print(f"Época {epoch:3d}/{epochs} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc Treino: {accuracy.item():.4f} | "
                      f"Acc Teste: {test_acc:.4f}")
            
            train_losses.append(loss.item())
            train_accuracies.append(accuracy.item())
        
        return train_losses, train_accuracies, test_accuracies
    
    def evaluate(self):
        """Avalia o modelo no conjunto de teste"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(self.X_test, "input_layer", depth=4)
            _, preds = torch.max(outputs, 1)
            accuracy = (preds == self.y_test).float().mean()
        return accuracy.item()
    
    def detailed_evaluation(self):
        """Avaliação detalhada do modelo"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(self.X_test, "input_layer", depth=4)
            _, preds = torch.max(outputs, 1)
            
            accuracy = accuracy_score(self.y_test.cpu(), preds.cpu())
            report = classification_report(
                self.y_test.cpu(), 
                preds.cpu(),
                target_names=['Setosa', 'Versicolor', 'Virginica']
            )
            
        print(f"\n{'='*50}")
        print("AVALIAÇÃO DETALHADA DO MODELO")
        print(f"{'='*50}")
        print(f"Acurácia total: {accuracy:.4f}")
        print("\nRelatório de classificação:")
        print(report)
        
        return accuracy, report
    
    def analyze_paths(self):
        """Analisa os caminhos mais utilizados"""
        print(f"\n{'='*50}")
        print("ANÁLISE DOS CAMINHOS MAIS UTILIZADOS")
        print(f"{'='*50}")
        
        # Contar frequência de camadas no histórico
        layer_counts = {}
        for layer in self.model.history:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        print("Frequência das camadas:")
        for layer, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {layer}: {count} vezes")
        
        # Mostrar probabilidades atuais
        print("\nProbabilidades atuais das camadas:")
        for layer in self.model.G.nodes():
            prob = self.model.G.nodes[layer]['prob']
            print(f"  {layer}: {prob:.4f}")
    
    def plot_results(self, train_losses, train_accuracies, test_accuracies):
        """Plota os resultados do treinamento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(train_losses, label='Loss de Treino', alpha=0.8)
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.set_title('Evolução da Loss durante o Treinamento')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        epochs = len(train_accuracies)
        test_epochs = np.linspace(0, epochs, len(test_accuracies))
        
        ax2.plot(train_accuracies, label='Acurácia Treino', alpha=0.8)
        ax2.plot(test_epochs, test_accuracies, label='Acurácia Teste', 
                marker='o', markersize=4, linewidth=2)
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Acurácia')
        ax2.set_title('Evolução da Acurácia durante o Treinamento')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('iris_classification_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def run_experiment():
    """Executa o experimento completo"""
    print("INICIANDO EXPERIMENTO DE CLASSIFICAÇÃO IRIS")
    print("=" * 50)
    
    # Criar e executar experimento
    experiment = IrisClassificationExperiment()
    
    # Treinar modelo
    train_losses, train_accuracies, test_accuracies = experiment.train(
        epochs=200, 
        depth=5
    )
    
    # Avaliação final
    final_accuracy, report = experiment.detailed_evaluation()
    
    # Análise dos caminhos
    experiment.analyze_paths()
    
    # Plotar resultados
    experiment.plot_results(train_losses, train_accuracies, test_accuracies)
    
    print(f"\n{'='*50}")
    print("EXPERIMENTO CONCLUÍDO!")
    print(f"Acurácia final: {final_accuracy:.4f}")
    print(f"{'='*50}")
    
    return experiment, final_accuracy

class DigitsClassificationExperiment:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Carregar e preparar dados
        self.load_digits_data()
        
        # Configurar modelo
        self.setup_model()
        
    def load_digits_data(self):
        """Carrega e prepara o dataset Digits"""
        print("Carregando dataset Digits...")
        
        # Carregar dados
        digits = load_digits()
        X, y = digits.data, digits.target
        
        print(f"Dataset Digits: {X.shape[0]} amostras, {X.shape[1]} features, {len(np.unique(y))} classes")
        
        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y  # 30% para teste
        )
        
        # Normalizar dados (importante para Digits)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Converter para tensores
        self.X_train = torch.FloatTensor(X_train).to(self.device)
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_train = torch.LongTensor(y_train).to(self.device)
        self.y_test = torch.LongTensor(y_test).to(self.device)
        
        # Salvar dados para visualização
        self.X_original = X
        self.y_original = y
        self.scaler = scaler
        
        print(f"Dados de treino: {self.X_train.shape}")
        print(f"Dados de teste: {self.X_test.shape}")
        print(f"Classes: {np.unique(y)}")
        
    def setup_model(self):
        """Configura o modelo BrainSys para Digits"""
        print("Configurando modelo BrainSys para Digits...")
        
        # Configurar Neuratron (maior para Digits)
        total_size = 200  # Maior que para Iris
        self.neura = Neuratron(total_size=(total_size, total_size))
        
        # Configurar otimizador
        self.optimizer = optim.Adam([
            {'params': self.neura.total_weight, 'lr': 0.005},  # LR menor
            {'params': self.neura.total_bias, 'lr': 0.005}
        ])
        
        # Configurar BrainSys para Digits
        input_size = 64  # 8x8 pixels = 64 features
        hidden_size = 32  # Camadas maiores
        output_size = 10  # 10 classes (dígitos 0-9)
        n_layers = 8      # Mais camadas
        
        self.model = BrainSys(
            neura=self.neura,
            learning_rate=0.005,
            size=(hidden_size, hidden_size),
            n_layers=n_layers,
            optimizer=self.optimizer
        ).to(self.device)
        
        # Alocar camada de entrada e saída especiais
        self.neura.allocate("input_layer", hidden_size, input_size)
        self.neura.allocate("output_layer", output_size, hidden_size)
        
        # Adicionar ao grafo
        self.model.G.add_node("input_layer", prob=1.0)
        self.model.G.add_node("output_layer", prob=1.0)
        
        # Conectar TODAS as camadas à camada de saída
        for layer in self.model.G.nodes():
            if layer != "output_layer":
                self.model.G.add_edge(layer, "output_layer", weight=1.0)
        
        print(f"Modelo configurado: {n_layers} camadas, {hidden_size} neurônios ocultos")
    
    def train(self, epochs=300, depth=6):
        """Treina o modelo no dataset Digits"""
        print(f"Iniciando treinamento por {epochs} épocas...")
        
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        best_accuracy = 0.0
        best_epoch = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Modo de treino
            self.model.train()
            
            # Forward pass
            outputs = self.model.forward(self.X_train, "input_layer", depth=depth)
            
            # Calcular perda
            loss = nn.CrossEntropyLoss()(outputs, self.y_train)
            
            # Backward pass
            self.model.backpropagation(outputs, self.y_train)
            
            # Calcular acurácia
            _, preds = torch.max(outputs, 1)
            accuracy = (preds == self.y_train).float().mean()
            
            # Avaliar no teste periodicamente
            if epoch % 20 == 0:
                test_acc = self.evaluate()
                test_accuracies.append(test_acc)
                
                # Salvar melhor modelo
                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    best_epoch = epoch
                    # Poderia salvar o modelo aqui
                
                elapsed = time.time() - start_time
                print(f"Época {epoch:3d}/{epochs} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc Treino: {accuracy.item():.4f} | "
                      f"Acc Teste: {test_acc:.4f} | "
                      f"Tempo: {elapsed:.1f}s")
            
            train_losses.append(loss.item())
            train_accuracies.append(accuracy.item())
        
        total_time = time.time() - start_time
        print(f"\nTreinamento concluído em {total_time:.1f} segundos")
        print(f"Melhor acurácia: {best_accuracy:.4f} na época {best_epoch}")
        
        return train_losses, train_accuracies, test_accuracies
    
    def evaluate(self):
        """Avalia o modelo no conjunto de teste"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(self.X_test, "input_layer", depth=6)
            _, preds = torch.max(outputs, 1)
            accuracy = (preds == self.y_test).float().mean()
        return accuracy.item()
    
    def detailed_evaluation(self):
        """Avaliação detalhada do modelo"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(self.X_test, "input_layer", depth=6)
            _, preds = torch.max(outputs, 1)
            
            accuracy = accuracy_score(self.y_test.cpu(), preds.cpu())
            report = classification_report(
                self.y_test.cpu(), 
                preds.cpu(),
                target_names=[str(i) for i in range(10)]
            )
            
            # Matriz de confusão
            cm = confusion_matrix(self.y_test.cpu(), preds.cpu())
            
        print(f"\n{'='*60}")
        print("AVALIAÇÃO DETALHADA - DATASET DIGITS")
        print(f"{'='*60}")
        print(f"Acurácia total: {accuracy:.4f}")
        print("\nRelatório de classificação:")
        print(report)
        
        # Plotar matriz de confusão
        self.plot_confusion_matrix(cm)
        
        return accuracy, report, cm
    
    def plot_confusion_matrix(self, cm):
        """Plota a matriz de confusão"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Matriz de Confusão - Classificação de Dígitos')
        plt.ylabel('Verdadeiro')
        plt.xlabel('Previsto')
        plt.tight_layout()
        plt.savefig('digits_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sample_images(self, num_samples=10):
        """Plota amostras de imagens do dataset"""
        plt.figure(figsize=(12, 6))
        for i in range(num_samples):
            plt.subplot(2, 5, i + 1)
            # Reconstruir imagem 8x8
            img = self.X_original[i].reshape(8, 8)
            plt.imshow(img, cmap='gray')
            plt.title(f'Dígito: {self.y_original[i]}')
            plt.axis('off')
        plt.suptitle('Amostras do Dataset Digits', fontsize=16)
        plt.tight_layout()
        plt.savefig('digits_samples.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_results(self, train_losses, train_accuracies, test_accuracies):
        """Plota os resultados do treinamento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(train_losses, label='Loss de Treino', alpha=0.7, color='red')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.set_title('Evolução da Loss durante o Treinamento')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        epochs = len(train_accuracies)
        test_epochs = np.linspace(0, epochs, len(test_accuracies))
        
        ax2.plot(train_accuracies, label='Acurácia Treino', alpha=0.7, color='blue')
        ax2.plot(test_epochs, test_accuracies, label='Acurácia Teste', 
                marker='o', markersize=4, linewidth=2, color='green')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Acurácia')
        ax2.set_title('Evolução da Acurácia durante o Treinamento')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('digits_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_performance(self):
        """Analisa performance por classe"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(self.X_test, "input_layer", depth=6)
            _, preds = torch.max(outputs, 1)
        
        # Calcular acurácia por classe
        class_accuracies = {}
        for digit in range(10):
            mask = self.y_test.cpu() == digit
            if mask.any():
                class_acc = (preds.cpu()[mask] == digit).float().mean()
                class_accuracies[digit] = class_acc.item()
        
        print(f"\n{'='*50}")
        print("ACURÁCIA POR CLASSE")
        print(f"{'='*50}")
        for digit, acc in sorted(class_accuracies.items()):
            print(f"Dígito {digit}: {acc:.4f}")

def run_digits_experiment():
    """Executa o experimento completo com Digits"""
    print("INICIANDO EXPERIMENTO DE CLASSIFICAÇÃO DE DÍGITOS")
    print("=" * 60)
    
    # Criar e executar experimento
    experiment = DigitsClassificationExperiment()
    
    # Mostrar amostras do dataset
    experiment.plot_sample_images()
    
    # Treinar modelo
    train_losses, train_accuracies, test_accuracies = experiment.train(
        epochs=300, 
        depth=6
    )
    
    # Avaliação final
    final_accuracy, report, cm = experiment.detailed_evaluation()
    
    # Análise de performance por classe
    experiment.analyze_performance()
    
    # Plotar resultados
    experiment.plot_results(train_losses, train_accuracies, test_accuracies)
    
    print(f"\n{'='*60}")
    print("EXPERIMENTO CONCLUÍDO!")
    print(f"Acurácia final nos dígitos: {final_accuracy:.4f}")
    print(f"{'='*60}")
    
    return experiment, final_accuracy

# Executar se este arquivo for chamado diretamente
def test_iris():
    # Verificar se temos GPU
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Executar experimento
    experiment, final_accuracy = run_experiment()
    
    # Exemplo de previsão única
    print("\nExemplo de previsão:")
    sample_idx = 0
    sample_input = experiment.X_test[sample_idx].unsqueeze(0)
    true_label = experiment.y_test[sample_idx]
    
    experiment.model.eval()
    with torch.no_grad():
        prediction = experiment.model.forward(sample_input, "input_layer", depth=5)
        predicted_class = torch.argmax(prediction).item()
    
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    print(f"Input: {sample_input.cpu().numpy().flatten()}")
    print(f"Verdadeiro: {class_names[true_label.item()]}")
    print(f"Previsto: {class_names[predicted_class]}")
    print(f"Confiança: {torch.softmax(prediction, 1).max().item():.4f}")

def test_digits():
    # Executar experimento
    experiment, final_accuracy = run_digits_experiment()
    
    # Exemplo de previsão
    print("\n📊 Exemplo de previsões:")
    for i in range(5):
        sample_idx = i
        sample_input = experiment.X_test[sample_idx].unsqueeze(0)
        true_label = experiment.y_test[sample_idx].item()
        
        experiment.model.eval()
        with torch.no_grad():
            prediction = experiment.model.forward(sample_input, "input_layer", depth=6)
            predicted_class = torch.argmax(prediction).item()
            confidence = torch.softmax(prediction, 1).max().item()
        
        status = "✅" if predicted_class == true_label else "❌"
        print(f"{status} Amostra {i}: Verdadeiro={true_label}, Previsto={predicted_class}, Confiança={confidence:.4f}")

if __name__ == '__main__':
    test_iris()
