import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, learning_curve

# Carregar o dataset
df_btc = pd.read_csv('datasets/coin_Bitcoin_PreProcessado.csv')

# Dividir os dados em variáveis independentes (X) e variável dependente (y)
X = df_btc.drop(['Close'], axis=1)  # Precisamos converter para o formato 2D
y = np.array(df_btc['Close'])

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Normalizar os dados
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Criar e treinar o modelo de rede neural
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
mlp.fit(X_train, y_train)

# Fazer previsões com o modelo
y_pred_train = mlp.predict(X_train)
y_pred_test = mlp.predict(X_test)

# Reverter a normalização das previsões
y_pred_train_original = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1))
y_pred_test_original = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1))
y_train_original = scaler_y.inverse_transform(y_train.reshape(-1, 1))
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Avaliar o modelo
mse_train = mean_squared_error(y_train_original, y_pred_train_original)
r2_train = r2_score(y_train_original, y_pred_train_original)
mse_test = mean_squared_error(y_test_original, y_pred_test_original)
r2_test = r2_score(y_test_original, y_pred_test_original)

print(f'MSE de treinamento: {mse_train}')
print(f'R² de treinamento: {r2_train}')
print(f'MSE de teste: {mse_test}')
print(f'R² de teste: {r2_test}')

# Validação cruzada
cv_scores = cross_val_score(mlp, X_train, y_train, cv=5, scoring='r2')
print(f'R² média da validação cruzada: {np.mean(cv_scores)}')

# Curvas de aprendizado
train_sizes, train_scores, test_scores = learning_curve(mlp, X_train, y_train, cv=5, scoring='r2', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score de treino")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de validação")
plt.xlabel("Tamanho do conjunto de treino")
plt.ylabel("Score R²")
plt.legend(loc="best")
plt.title("Curvas de Aprendizado")

# Gráfico de Resíduos
plt.subplot(1, 2, 2)
plt.scatter(y_test_original, y_test_original - y_pred_test_original, color='blue', edgecolor='k', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Valores Reais")
plt.ylabel("Resíduos")
plt.title("Gráfico de Resíduos")

plt.tight_layout()
plt.show()


# Plotar os resultados
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train_original, y_pred_train_original, color='blue', edgecolor='k', alpha=0.7)
plt.plot([min(y_train_original), max(y_train_original)], [min(y_train_original), max(y_train_original)], color='red', linestyle='--')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')
plt.title('Dados de Treinamento')

plt.subplot(1, 2, 2)
plt.scatter(y_test_original, y_pred_test_original, color='blue', edgecolor='k', alpha=0.7)
plt.plot([min(y_test_original), max(y_test_original)], [min(y_test_original), max(y_test_original)], color='red', linestyle='--')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')
plt.title('Dados de Teste')

plt.tight_layout()
plt.show()