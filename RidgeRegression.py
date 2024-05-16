import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df_btc = pd.read_csv('datasets/coin_Bitcoin_PreProcessado.csv')

    x = df_btc.drop(['Close'], axis=1)
    y = np.asarray(df_btc['Close'])

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

    # Aplicando Ridge Regression
    ridge_regressor = Ridge(alpha=50.0)  # Você pode ajustar o valor de alpha para regularizar mais ou menos
    ridge_regressor.fit(X_train, y_train)

    # Fazendo previsões com o conjunto de teste
    y_pred = ridge_regressor.predict(X_test)

    # Calculando o erro quadrático médio
    mse = mean_squared_error(y_test, y_pred)
    print("Erro quadrático médio (MSE):", mse)

    # Calculando o coeficiente de determinação (R²)
    r2 = r2_score(y_test, y_pred)
    print("Coeficiente de determinação (R²):", r2)

    # Coeficientes da regressão
    print("Coeficientes:", ridge_regressor.coef_)

    # Plot 1: Valores previstos vs. valores reais
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Previsto")
    plt.title("Valores Previstos vs. Valores Reais")
    plt.show()

    # Ordenar os dados por data para o próximo plot
    X_test_sorted = X_test.copy()
    X_test_sorted['Real'] = y_test
    X_test_sorted['Pred'] = y_pred
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
    residuos = y_test - y_pred

    # Plot 3: Análise Residual
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test['Date'], residuos, alpha=0.9)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Data")
    plt.ylabel("Resíduos")
    plt.title("Análise Residual")
    plt.show()