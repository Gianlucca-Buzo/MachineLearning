import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats,linalg
from shap.maskers import Independent
import lime.lime_tabular
import warnings
from lime import submodular_pick

def ReturnPValue(model,X,Y):
    YHat = model.predict(X)
    n,k = X.shape
    sse = np.sum(np.square(YHat-Y),axis=0)
    x = np.hstack((np.ones((n,1)),np.matrix(X)))
    df = float(n-k-1)
    sampleVar = sse/df
    sampleVarianceX = x.T*x
    covarianceMatrix = linalg.sqrtm(sampleVar*sampleVarianceX.I)
    se = covarianceMatrix.diagonal()[1:]
    betasTstat = np.zeros(len(se))
    for i in range(len(se)):
        betasTstat[i] = model.coef_[i]/se[i]
    betasPvalue = 1- stats.t.cdf(abs(betasTstat),df)
    return betasPvalue

def AdjustedRSquare(model,X,Y):
    YHat = model.predict(X)
    n,k = X.shape
    sse = np.sum(np.square(YHat - Y), axis=0)  # sum of suare error
    sst = np.sum(np.square(Y - np.mean(Y)), axis=0)  # sum of square total
    R2 = 1 - sse / sst  # explained sum of squares
    adjR2 = R2 - (1 - R2) * (float(k) / (n - k - 1))
    return adjR2, R2

def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in
    range(X.shape[1])]
    return (vif)

def print_coeficientes(y_train,y_train_pred,y_test,regressor):
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

def print_plots(y_test,y_test_pred,X_test,):
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

def print_vif(df_btc,x,y_train,X_train,regressor):
    # Explicabilidade
    sns.pairplot(df_btc[['Date', 'High', 'Low', 'Volume']])

    corrl = (df_btc[['Date', 'High', 'Low', 'Volume']]).corr()
    print("#### CORRL VALUES ####")
    print(corrl)
    print("#### CORRL VALUES END ####")

    vif_df = calc_vif(x)
    vif_df.sort_values(by='VIF', ascending=False).head()

    print("#### VIF VALUES ####")
    print(vif_df)
    print("#### VIF VALUES END ####")

    resultsDF = pd.DataFrame()
    resultsDF['Variables'] = pd.Series(X_train.columns)
    resultsDF['coefficients'] = pd.Series(np.round(regressor.coef_, 10))
    resultsDF['p_value'] = pd.Series(np.round(ReturnPValue(regressor, X_train,
                                                           y_train), 10))
    resultsDF.sort_values(by='p_value', ascending=False)
    print("#### RESULTS VALUES ####")
    print(resultsDF)
    print("#### RESULTS END VALUES ####")

def print_validacao_cruzada(x,y):
    # Validação cruzada
    cv_scores_mse = cross_val_score(regressor, x, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores_r2 = cross_val_score(regressor, x, y, cv=5, scoring='r2')

    print("MSE média da validação cruzada:", -cv_scores_mse.mean())
    print("R² médio da validação cruzada:", cv_scores_r2.mean())

def calc_shap(X_train,regressor):
    # calculo de valores SHAP
    background = shap.maskers.Independent(X_train, max_samples=2000)
    explainer = shap.Explainer(regressor, background)
    shap_values = explainer(X_train)

    shap.plots.waterfall(shap_values[60], max_display=30)

    print("#### SHAP VALUES ####")
    print(pd.DataFrame(np.round(shap_values.values, 3)).head(3))

    print("\n#### SHAP BASE VALUES #### ")
    print(pd.DataFrame(np.round(shap_values.base_values, 3)).head(3))

def calc_lime(X_train,regressor,X_test):
    columns = list(X_train.columns)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, mode='regression',
                                                            feature_names=columns,
                                                            class_names=['column1', 'column2', 'column3', 'column4',
                                                                         'column5'],
                                                            verbose=True)
    i = 60
    x_test_instance = X_test.iloc[[i]]  # Garantir que a instância tenha a forma correta como DataFrame
    exp = lime_explainer.explain_instance(x_test_instance.values[0], regressor.predict, num_features=5)
    exp.show_in_notebook(show_table=True)
    print(exp.as_list())

    sp_obj = submodular_pick.SubmodularPick(lime_explainer, np.array(X_train),
                                            regressor.predict,
                                            num_features=14,
                                            num_exps_desired=10)
    j = 1
    for exp in sp_obj.sp_explanations:
        exp.save_to_file("/output_lime/out-" + str(j) + ".html")
        j = j + 1


# Função principal
if __name__ == '__main__':
    # Carregar o dataset
    df_btc = pd.read_csv('datasets/coin_Bitcoin_PreProcessado.csv')

    # Separar variáveis independentes (X) e dependente (y)
    x = df_btc.drop(['Close'], axis=1)
    y = np.asarray(df_btc['Close'])

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=.5, random_state=42)

    # Treinar o modelo de regressão linear
    regressor = LinearRegression()
    regressor.fit(X_train.values, y_train)

    # Fazer previsões com o conjunto de treinamento
    y_train_pred = regressor.predict(X_train.values)

    # Fazer previsões com o conjunto de teste
    y_test_pred = regressor.predict(X_test.values)

    # print_coeficientes(y_train,y_train_pred,y_test,regressor)
    # print_validacao_cruzada(x,y)
    # print_vif(df_btc,x,y_train,X_train,regressor)
    # print_plots(y_test,y_test_pred,X_test)
    #
    # calc_shap(X_train,regressor)
    # calc_lime(X_train,regressor,X_test)

