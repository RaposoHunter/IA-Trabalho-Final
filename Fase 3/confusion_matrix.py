import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix

# Lê os dados do arquivo do Excel
dados = pd.read_excel('dados tratados.xlsx', sheet_name='Dados finais')

# Separa as colunas de entrada (features) e a coluna alvo (target)
features = dados.iloc[:, :-1]
target = dados.iloc[:, -1]

# Criar a arquitetura da rede neural
model = Sequential()
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compilar o modelo
optimizer = SGD(learning_rate=0.4, momentum=0.6)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',  # Neste exemplo, usamos a função de perda para classificação binária
              metrics=['accuracy'])

# Treinar a rede neural
model.fit(features, target, epochs=50, verbose=0)

# Fazer previsões usando o modelo
predictions = model.predict(features)
predictions = (predictions > 0.5).astype(int)  # Convertendo probabilidades para rótulos binários (0 ou 1)

# Criar a matriz de confusão
cm = confusion_matrix(target, predictions)

# Calcular as porcentagens
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Criar um heatmap da matriz de confusão para visualização em porcentagem
labels = ['0', '1']  # Rótulos da classe (0 e 1)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Previsão')
plt.ylabel('True')
plt.title('Matriz de confusão (%)')
plt.show()