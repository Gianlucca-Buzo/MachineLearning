import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df_btc = pd.read_csv('BTC-USD.csv')

    df_btc['DateTime'] = pd.to_datetime(df_btc['Date'])
    df_btc['DateTime'] = df_btc['DateTime'].map(dt.datetime.toordinal)

    x = np.asarray(df_btc['DateTime']).reshape(-1, 1)  # Redimensiona para 2D
    y = np.asarray(df_btc['Close'])

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=.7, random_state=42)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Fazendo previsões com o conjunto de teste
    y_pred = regressor.predict(X_test)

    # Calculando o erro quadrático médio
    mse = mean_squared_error(y_test, y_pred)
    print("Erro quadrático médio (MSE):", mse)

    # Calculando o coeficiente de determinação (R²)
    r2 = r2_score(y_test, y_pred)
    print("Coeficiente de determinação (R²):", r2)

    # Coeficientes da regressão
    print("Coeficientes:", regressor.coef_)

    # Plotando os valores previstos vs. valores reais
    plt.scatter(y_test, regressor.predict(X_test))
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Previsto")
    plt.title("Valores Previstos vs. Valores Reais")
    plt.show()

    # Plotando a relação entre a variável de entrada e a variável de saída
    plt.scatter(X_test, y_test)
    plt.plot(X_test, regressor.predict(X_test), color='red')
    plt.xlabel("Data")
    plt.ylabel("Preço de Fechamento")
    plt.title("Regressão Linear")
    plt.show()

    # Calculando os resíduos
    residuos = y_test - regressor.predict(X_test)

    # Plotando os resíduos
    plt.scatter(X_test, residuos)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Data")
    plt.ylabel("Resíduos")
    plt.title("Análise Residual")
    plt.show()