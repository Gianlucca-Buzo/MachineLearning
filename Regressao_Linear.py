import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Função principal
if __name__ == '__main__':
    # Carregar o dataset
    df_btc = pd.read_csv('datasets/coin_Bitcoin_PreProcessado.csv')

    # Separar variáveis independentes (X) e dependente (y)
    x = df_btc.drop(['Close','Volume'], axis=1)
    y = np.asarray(df_btc['Close'])

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=.5, random_state=42)

    # Treinar o modelo de regressão linear
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Fazer previsões com o conjunto de treinamento
    y_train_pred = regressor.predict(X_train)

    # Fazer previsões com o conjunto de teste
    y_test_pred = regressor.predict(X_test)

    # Calculando o erro quadrático médio para o conjunto de treinamento
    mse_train = mean_squared_error(y_train, y_train_pred)
    print("Erro quadrático médio (MSE) - Treinamento:", mse_train)

    # Calculando o coeficiente de determinação (R²) para o conjunto de treinamento
    r2_train = r2_score(y_train, y_train_pred)
    print("Coeficiente de determinação (R²) - Treinamento:", r2_train)

    # Calculando o erro quadrático médio para o conjunto de teste
    mse_test = mean_squared_error(y_test, y_test_pred)
    print("Erro quadrático médio (MSE) - Teste:", mse_test)

    # Calculando o coeficiente de determinação (R²) para o conjunto de teste
    r2_test = r2_score(y_test, y_test_pred)
    print("Coeficiente de determinação (R²) - Teste:", r2_test)

    # Coeficientes da regressão
    print("Coeficientes da regressão:", regressor.coef_)

    # Plot 1: Valores previstos vs. valores reais
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Previsto")
    plt.title("Valores Previstos vs. Valores Reais")
    plt.show()

    # Ordenar os dados por data para o próximo plot
    X_test_sorted = X_test.copy()
    X_test_sorted['Real'] = y_test
    X_test_sorted['Pred'] = y_test_pred
    X_test_sorted = X_test_sorted.sort_values(by='Date')

    # Plot 2: Histórico de valores reais e previstos
    plt.figure(figsize=(12, 6))
    plt.plot(X_test_sorted['Date'], X_test_sorted['Real'], label='Real', linestyle='dotted', marker='o', alpha=0.75)
    plt.plot(X_test_sorted['Date'], X_test_sorted['Pred'], color='red', label='Previsto', linestyle='-', alpha=0.75)
    plt.xlabel("Data")
    plt.ylabel("Preço de Fechamento")
    plt.title("Histórico de Valores Reais e Previstos")
    plt.legend()
    plt.show()

    # Calculando os resíduos
    residuos = y_test - y_test_pred

    # Plot 3: Análise Residual
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test['Date'], residuos, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Data")
    plt.ylabel("Resíduos")
    plt.title("Análise Residual")
    plt.show()

    # Validação cruzada
    cv_scores_mse = cross_val_score(regressor, x, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores_r2 = cross_val_score(regressor, x, y, cv=5, scoring='r2')

    print("MSE média da validação cruzada:", -cv_scores_mse.mean())
    print("R² médio da validação cruzada:", cv_scores_r2.mean())
