import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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

    # Substituir regressor linear por árvore de decisão
    regressor = DecisionTreeRegressor(random_state=42)
    regressor.fit(X_train, y_train)

    # Fazendo previsões com o conjunto de teste
    y_pred = regressor.predict(X_test)

    # Calculando o erro quadrático médio
    mse = mean_squared_error(y_test, y_pred)
    print("Erro quadrático médio (MSE):", mse)

    # Calculando o coeficiente de determinação (R²)
    r2 = r2_score(y_test, y_pred)
    print("Coeficiente de determinação (R²):", r2)

    # Ordenando os pontos de dados por sua variável de entrada para melhor visualização
    sorted_indices = np.argsort(X_test[:, 0])
    X_test_sorted = X_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    # Convertendo a data de volta para o formato "yyyy-mm-dd"
    datas_convertidas = [dt.datetime.fromordinal(int(data)).strftime('%Y-%m-%d') for data in X_test_sorted[:, 0]]

    # Plotando a relação entre a variável de entrada e a variável de saída
    plt.scatter(datas_convertidas, y_test[sorted_indices])
    plt.plot(datas_convertidas, y_pred_sorted, color='red')
    plt.xlabel("Data")
    plt.ylabel("Preço de Fechamento")
    plt.title("Árvore de Decisão")
    plt.xticks(np.arange(0, len(datas_convertidas), 100), rotation=45)  # Ajusta a frequência e rotação dos rótulos do eixo x
    plt.show()

    # Calculando os resíduos
    residuos = y_test - y_pred

    # Plotando os resíduos
    plt.scatter(X_test, residuos)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Data")
    plt.ylabel("Resíduos")
    plt.title("Análise Residual")
    plt.show()