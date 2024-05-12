from sklearn import svm
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE

data = pd.read_csv('BTC-USD_PreProcessado.csv')

# Dividir os dados em variáveis independentes (X) e variável dependente (y)
X = np.array(data['DateTime']).reshape(-1, 1)  # Precisamos converter para o formato 2D
y = np.array(data['Close'])

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

stdsc = StandardScaler()
x_train_std = stdsc.fit_transform(X_train)
x_test_std = stdsc.transform(X_test)

svm_model = svm.SVR(kernel='rbf', C=1000000, gamma='scale')
scores = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']
resultados = cross_validate(svm_model, x_train_std, y_train, cv=10, scoring=scores)
pd.DataFrame(resultados).mean()

modelo_treinado_svm = svm_model.fit(x_train_std, y_train)
y_pred = modelo_treinado_svm.predict(x_test_std)

# Plotar os resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Dados')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='gray', linestyle='--')
plt.xlabel('Preço Real')
plt.ylabel('Preço Previsto')
plt.title('Comparação entre Preços Reais e Previstos')
plt.legend()
plt.show()

mse = np.sqrt(MSE(y_test, y_pred))
print("Erro quadrático médio (MSE):", mse)
