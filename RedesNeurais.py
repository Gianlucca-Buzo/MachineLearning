import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, learning_curve
from lime import lime_tabular
import shap
import datetime as dt

# Funções de criação de features (podem ser movidas para um utilitário depois)
def criar_features_defasadas(df, nome_coluna_alvo, num_lags):
    for i in range(1, num_lags + 1):
        df[f'Lag_{i}_{nome_coluna_alvo}'] = df[nome_coluna_alvo].shift(i) # Nome do lag mais específico
    return df

def criar_features_data(df, nome_coluna_data):
    df[nome_coluna_data] = pd.to_datetime(df[nome_coluna_data])
    df['Year'] = df[nome_coluna_data].dt.year
    df['Month'] = df[nome_coluna_data].dt.month
    df['Day'] = df[nome_coluna_data].dt.day
    df['DayOfWeek'] = df[nome_coluna_data].dt.dayofweek
    return df

# Configurações Globais
ARQUIVO_DADOS = 'BTC-USD.csv' # Usando o mesmo CSV base
COLUNA_ALVO = 'Close'
COLUNA_DATA = 'Date'
NUM_LAGS = 5
PERCENTUAL_TREINO = 0.8

if __name__ == '__main__':
    # Carregar o dataset
    df_btc = pd.read_csv(ARQUIVO_DADOS)

    # ---- Engenharia de Features ----
    df_btc = criar_features_data(df_btc, COLUNA_DATA)
    # Criar lags para Open, High, Low, Close, Volume se existirem
    cols_para_lags = ['Open', 'High', 'Low', COLUNA_ALVO, 'Volume']
    for col in cols_para_lags:
        if col in df_btc.columns:
            df_btc = criar_features_defasadas(df_btc, col, NUM_LAGS)
        else:
            print(f"Aviso: Coluna {col} não encontrada para criar lags.")

    # Remover linhas com NaN geradas pelos lags
    df_btc = df_btc.dropna()

    # Selecionar features (X) e alvo (y)
    features = [col for col in df_btc.columns if col not in [COLUNA_ALVO, COLUNA_DATA]]
    X = df_btc[features]
    y = df_btc[COLUNA_ALVO]

    # ---- Divisão de Dados Cronológica ----
    tamanho_treino = int(len(df_btc) * PERCENTUAL_TREINO)
    X_train_df, X_test_df = X.iloc[:tamanho_treino], X.iloc[tamanho_treino:]
    y_train_series, y_test_series = y.iloc[:tamanho_treino], y.iloc[tamanho_treino:]

    # Manter nomes de colunas para XAI
    feature_names = X_train_df.columns.tolist()

    # ---- Normalizar os dados ----
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train_df)
    X_test = scaler_X.transform(X_test_df)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train_series.values.reshape(-1, 1)).flatten()
    # y_test não é transformado aqui, pois será usado na escala original para avaliação
    # e as predições serão revertidas para a escala original.

    # ---- Modelo: Rede Neural ----
    # Hiperparâmetros podem ser otimizados com GridSearchCV ou RandomizedSearchCV
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42, max_iter=500, early_stopping=True, n_iter_no_change=10)
    print(f"Treinando modelo: {mlp.__class__.__name__}")
    mlp.fit(X_train, y_train)

    # Fazer previsões com o modelo (as previsões estarão na escala normalizada de y)
    # y_pred_train_scaled = mlp.predict(X_train)
    y_pred_test_scaled = mlp.predict(X_test)

    # Reverter a normalização das previsões para a escala original
    # y_pred_train_original = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()
    y_pred_test_original = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()
    
    # y_train_original = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_train_original = y_train_series.values # Já está na escala original
    y_test_original = y_test_series.values # Já está na escala original

    # ---- Avaliação do Modelo ----
    # mse_train = mean_squared_error(y_train_original, y_pred_train_original)
    # r2_train = r2_score(y_train_original, y_pred_train_original)
    # mae_train = mean_absolute_error(y_train_original, y_pred_train_original)
    
    mse_test = mean_squared_error(y_test_original, y_pred_test_original)
    r2_test = r2_score(y_test_original, y_pred_test_original)
    mae_test = mean_absolute_error(y_test_original, y_pred_test_original)

    print(f"---- Resultados para {mlp.__class__.__name__} ----")
    # print(f'MSE de treinamento: {mse_train}')
    # print(f'R² de treinamento: {r2_train}')
    # print(f'MAE de treinamento: {mae_train}')
    print(f'MSE de teste: {mse_test}')
    print(f'R² de teste: {r2_test}')
    print(f'MAE de teste: {mae_test}')

    # ---- XAI: LIME ----
    print("\n---- Gerando explicações LIME ----")
    explainer_lime = lime_tabular.LimeTabularExplainer(
        training_data=X_train, # Usar dados normalizados para LIME, pois o modelo foi treinado com eles
        feature_names=feature_names,
        class_names=[COLUNA_ALVO],
        mode='regression',
        random_state=42
    )

    # Explicar uma instância específica (por exemplo, a primeira instância de teste)
    i = 0
    exp_lime = explainer_lime.explain_instance(X_test[i], mlp.predict, num_features=len(feature_names))
    lime_file = f'lime_explanation_{mlp.__class__.__name__}_instance_{i}.html'
    exp_lime.save_to_file(lime_file)
    print(f"Explicação LIME para a instância {i} salva como {lime_file}")

    # ---- XAI: SHAP ----
    print("\n---- Gerando explicações SHAP ----")
    # SHAP com KernelExplainer para modelos como MLPRegressor
    # Usar uma amostra dos dados de treino como background para KernelExplainer
    # shap.sample(X_train, 100) cria uma amostra de 100 pontos se X_train for grande
    background_data_shap = shap.sample(X_train, min(100, X_train.shape[0])) 
    explainer_shap = shap.KernelExplainer(mlp.predict, background_data_shap)
    shap_values = explainer_shap.shap_values(X_test) # Pode demorar um pouco

    plt.figure()
    shap.summary_plot(shap_values, X_test_df, feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary Plot - {mlp.__class__.__name__}')
    shap_summary_file = f'shap_summary_{mlp.__class__.__name__}.png'
    plt.savefig(shap_summary_file, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"SHAP summary plot salvo como {shap_summary_file}")

    # ---- Visualizações ----
    datas_plot_teste = df_btc[COLUNA_DATA].iloc[tamanho_treino:]

    plt.figure(figsize=(12, 6))
    plt.plot(datas_plot_teste, y_test_original, label='Real', color='blue', marker='.')
    plt.plot(datas_plot_teste, y_pred_test_original, label='Predito', color='red', linestyle='--')
    plt.xlabel("Data")
    plt.ylabel(f"Preço de Fechamento {COLUNA_ALVO}")
    plt.title(f"Previsão de Preços com {mlp.__class__.__name__}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    pred_actual_file = f'prediction_vs_actual_{mlp.__class__.__name__}.png'
    plt.savefig(pred_actual_file)
    plt.show()
    plt.close()
    print(f"Gráfico de previsão salvo como {pred_actual_file}")

    # Gráfico de Resíduos
    residuos = y_test_original - y_pred_test_original
    plt.figure(figsize=(10, 6))
    plt.scatter(datas_plot_teste, residuos, color='blue', edgecolor='k', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Data")
    plt.ylabel("Resíduos")
    plt.title(f"Análise Residual - {mlp.__class__.__name__}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    residuals_file = f'residuals_plot_{mlp.__class__.__name__}.png'
    plt.savefig(residuals_file)
    plt.show()
    plt.close()
    print(f"Gráfico de resíduos salvo como {residuals_file}")

    # Curvas de aprendizado e validação cruzada podem ser reativadas se necessário
    # print("\nCalculando Validação Cruzada e Curvas de Aprendizado (pode demorar)...")
    # cv_scores = cross_val_score(mlp, X_train, y_train, cv=3, scoring='r2', n_jobs=-1) # cv=3 para rapidez
    # print(f'R² média da validação cruzada: {np.mean(cv_scores)}')

    # train_sizes, train_scores, test_scores_cv = learning_curve(mlp, X_train, y_train, cv=3, scoring='r2', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
    # train_scores_mean = np.mean(train_scores, axis=1)
    # test_scores_mean_cv = np.mean(test_scores_cv, axis=1)
    # plt.figure()
    # plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score de treino")
    # plt.plot(train_sizes, test_scores_mean_cv, 'o-', color="g", label="Score de validação cruzada")
    # plt.xlabel("Tamanho do conjunto de treino")
    # plt.ylabel("Score R²")
    # plt.title("Curvas de Aprendizado")
    # plt.legend(loc="best")
    # plt.savefig(f'learning_curve_{mlp.__class__.__name__}.png')
    # plt.show()
    # plt.close()