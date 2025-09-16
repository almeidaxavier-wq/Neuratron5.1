# config.py - Configurações do experimento
EXPERIMENT_CONFIG = {
    'dataset': 'iris',
    'input_size': 4,
    'hidden_size': 10,
    'output_size': 3,
    'n_layers': 6,
    'learning_rate': 0.01,
    'epochs': 200,
    'depth': 5,
    'test_size': 0.2,
    'random_state': 42
}

# Nomes das classes do Iris
CLASS_NAMES = {
    0: 'Iris Setosa',
    1: 'Iris Versicolor', 
    2: 'Iris Virginica'
}