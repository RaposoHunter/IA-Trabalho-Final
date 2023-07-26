import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Lê os dados do arquivo do Excel
dados = pd.read_excel('dados tratados.xlsx', sheet_name='Dados finais')

# Separa as colunas de entrada (features) e a coluna alvo (target)
features = dados.iloc[:, :-1]
target = dados.iloc[:, -1]

# Definir as listas de valores a serem testados
camadas = [1, 2, 3, 4, 5]
neuronios = [2, 4, 8, 16, 32, 64, 128]
learning_rates = [0.3, 0.4, 0.5, 0.6, 0.7]
momentums = [0.3, 0.4, 0.5, 0.6, 0.7]

with open("data.csv", "w", encoding='utf8') as arquivo:
    arquivo.write("Nome da rede,Acuracia,Camadas,neuronios,learning_rate,momentum\n")

# Loop sobre as configurações de hiperparâmetros
for camada in camadas:
    for neuronio in neuronios:
        for lr in learning_rates:
            for momentum in momentums:
                # Criar a arquitetura da rede neural
                model = Sequential()
                for _ in range(camada):
                    model.add(Dense(neuronio, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))  # Saída binária (0 ou 1)

                # Compilar o modelo
                optimizer = SGD(learning_rate=lr, momentum=momentum)
                model.compile(optimizer=optimizer,
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

                # Treinar a rede neural
                model.fit(features, target, epochs=50, verbose=0)

                # Avaliar a precisão do modelo nos dados de treino
                loss, accuracy = model.evaluate(features, target, verbose=0)
                with open("data.csv", "a", encoding='utf8') as arquivo:
                    # Escrever o conteúdo no arquivo
                    arquivo.write(f"C{camada}N{neuronio}L{str(lr*10).replace('.0','')}M{str(momentum*10).replace('.0','')},{accuracy},{camada},{neuronio},{lr},{momentum}\n")