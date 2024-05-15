import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Carregar os dados
df_btc = pd.read_csv('BTC-USD.csv')

# Converter a coluna de data para o formato datetime
df_btc['DateTime'] = pd.to_datetime(df_btc['Date'])

# Mapear a coluna de data para um número ordinal
df_btc['DateTime'] = df_btc['DateTime'].map(dt.datetime.toordinal)

# Dividir os dados em variáveis independentes (X) e variável dependente (y)
x = np.asarray(df_btc['DateTime']).reshape(-1, 1)  # Redimensionar para 2D
y = np.asarray(df_btc['Close'])

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=.7, random_state=42)

# Instanciar e treinar o modelo RandomForestRegressor
regressor = RandomForestRegressor(random_state=0, n_estimators=100, max_depth=5)
regressor.fit(X_train, y_train)

# Fazer previsões com o conjunto de teste
y_pred = regressor.predict(X_test)

# Calcular o erro quadrático médio
mse = mean_squared_error(y_test, y_pred)
print("Erro quadrático médio (MSE):", mse)

# Calcular o coeficiente de determinação (R²)
r2 = r2_score(y_test, y_pred)
print("Coeficiente de determinação (R²):", r2)

# Ordenar os pontos de dados por sua variável de entrada para melhor visualização
sorted_indices = np.argsort(X_test[:, 0])
X_test_sorted = X_test[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

# Converter a data de volta para o formato "yyyy-mm-dd"
datas_convertidas = [dt.datetime.fromordinal(int(data)).strftime('%Y-%m-%d') for data in X_test_sorted[:, 0]]

# Plotar a relação entre a variável de entrada e a variável de saída
plt.scatter(datas_convertidas, y_test[sorted_indices])
plt.plot(datas_convertidas, y_pred_sorted, color='red')
plt.xlabel("Data")
plt.ylabel("Preço de Fechamento")
plt.title("Random Forest Regressor")
plt.xticks(np.arange(0, len(datas_convertidas), 100), rotation=45)  # Ajustar a frequência e rotação dos rótulos do eixo x
plt.show()

# Calcular os resíduos
residuos = y_test - y_pred

# Plotar os resíduos
plt.scatter(datas_convertidas, residuos)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Data")
plt.ylabel("Resíduos")
plt.title("Análise Residual")
plt.xticks(np.arange(0, len(datas_convertidas), 100), rotation=45)  # Ajustar a frequência e rotação dos rótulos do eixo x
plt.show()