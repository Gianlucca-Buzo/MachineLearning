import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
# from sklearn.model_selection import GridSearchCV # Comentado para execução mais rápida
import shap
from lime import lime_tabular
from outputs.utils import criar_features_defasadas, criar_features_data # Importando de utils

print("DEBUG: Módulos importados - SVM.py")

# Carregar os dados
print("DEBUG: Carregando dados...")
try:
    df_original = pd.read_csv("BTC-USD.csv")
    print("DEBUG: Dados crus carregados:")
    print(df_original.head())
except FileNotFoundError:
    print("ERRO: Arquivo BTC-USD.csv não encontrado.")
    exit()

df = df_original.copy()

# Convertendo a coluna 'Date' para datetime
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
else:
    print("ERRO: Coluna 'Date' não encontrada.")
    exit()

# Engenharia de Features (Cenário A: Prever Close(D+1) usando dados até D)
print("DEBUG: Iniciando engenharia de features...")
N_LAGS = 5
COLUNAS_PARA_LAGS = ['Open', 'High', 'Low', 'Close', 'Volume']

# Criar o alvo (Close do dia seguinte)
df['Target_Close_D+1'] = df['Close'].shift(-1)

# Criar lags para as colunas especificadas
# Os nomes dos lags serão como 'LagOpen_1_Open', 'LagClose_1_Close', etc.
for coluna in COLUNAS_PARA_LAGS:
    df = criar_features_defasadas(df, coluna, N_LAGS, prefixo_coluna_lag=f'Lag{coluna}_')

# Criar features de data
df = criar_features_data(df, 'Date')

# Remover linhas com NaN resultantes da criação de lags e do target_shift
print(f"DEBUG: Linhas antes do dropna: {len(df)}")
df.dropna(inplace=True)
print(f"DEBUG: Linhas após o dropna: {len(df)}")

if df.empty:
    print("ERRO: DataFrame vazio após dropna. Verifique a criação de lags e do target.")
    exit()

# Seleção de Features (X) e Alvo (y)
features_potenciais = [
    'Open', 'High', 'Low', 'Close', 'Volume',  # Dados do dia D
    'Year', 'Month', 'Day', 'DayOfWeek'  # Features de data do dia D
]
# Adicionar colunas de lag à lista de features
for coluna_base in COLUNAS_PARA_LAGS:
    for i in range(1, N_LAGS + 1):
        features_potenciais.append(f'Lag{coluna_base}_{i}_{coluna_base}')

# Garantir que todas as features potenciais existem no DataFrame após o dropna
features_finais = [f for f in features_potenciais if f in df.columns]

# Remover colunas que não devem ser features
features_a_remover = ['Date', 'Adj Close', 'Target_Close_D+1']
features_finais = [f for f in features_finais if f not in features_a_remover and f in df.columns]

print(f"DEBUG: Features finais selecionadas para o modelo SVM: {features_finais}")

X = df[features_finais]
y = df['Target_Close_D+1']
datas_plot = df['Date']  # Para os gráficos

# Divisão dos dados em treino e teste (CRONOLÓGICO)
TEST_SIZE = 0.2
X_train, X_test, y_train, y_test, datas_train, datas_test = train_test_split(
    X, y, datas_plot, test_size=TEST_SIZE, shuffle=False
)

print(f"DEBUG: Tamanho do conjunto de treino: {X_train.shape[0]}")
print(f"DEBUG: Tamanho do conjunto de teste: {X_test.shape[0]}")

if X_train.empty or X_test.empty:
    print("ERRO: Conjunto de treino ou teste vazio. Verifique o tamanho dos dados e o TEST_SIZE.")
    exit()

# Normalização dos dados (importante para SVM)
print("DEBUG: Normalizando os dados...")
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
# Não vamos escalar y por enquanto para simplificar a interpretação e a inversão.

# Treinamento do Modelo SVM
print("DEBUG: Treinando o modelo SVR...")
# Parâmetros de exemplo. GridSearchCV pode ser usado para otimização (código comentado abaixo)
modelo_svr = SVR(kernel='rbf', C=100, gamma=0.1)
modelo_svr.fit(X_train_scaled, y_train)
print("DEBUG: Modelo SVR treinado.")
modelo = modelo_svr # Usar 'modelo' como nome genérico

# # Opcional: GridSearchCV para SVR (pode ser demorado)
# print("DEBUG: Configurando GridSearchCV para SVR...")
# param_grid_svr = {
#     'C': [10, 100],
#     'gamma': [0.01, 0.1, 'scale'],
#     'kernel': ['rbf']
# }
# grid_search_svr = GridSearchCV(SVR(), param_grid_svr, cv=2, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
# grid_search_svr.fit(X_train_scaled, y_train)
# print(f"DEBUG: Melhores parâmetros para SVR: {grid_search_svr.best_params_}")
# modelo = grid_search_svr.best_estimator_

# Previsões
print("DEBUG: Fazendo previsões com SVR...")
previsoes = modelo.predict(X_test_scaled)
# Nenhuma inversão de y necessária pois não foi escalado

# Avaliação do Modelo
print("DEBUG: Avaliando o modelo SVR...")
mse = mean_squared_error(y_test, previsoes)
mae = mean_absolute_error(y_test, previsoes)
r2 = r2_score(y_test, previsoes)

print(f"\n--- Resultados do Modelo ({modelo.__class__.__name__}) ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Plot: Previsões vs Valores Reais
print("DEBUG: Gerando gráfico Previsões vs Reais (SVR)...")
plt.figure(figsize=(15, 7))
plt.plot(datas_test.values, y_test.values, label='Valores Reais', color='blue', marker='.', linestyle='-')
plt.plot(datas_test.values, previsoes, label=f'Previsões {modelo.__class__.__name__}', color='green', marker='.', linestyle='--')
plt.title(f'Previsão de Preço (Close D+1) - {modelo.__class__.__name__}')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'prediction_vs_actual_{modelo.__class__.__name__}.png')
plt.close()
print(f"DEBUG: Gráfico 'prediction_vs_actual_{modelo.__class__.__name__}.png' salvo.")

# Plot: Resíduos
print("DEBUG: Gerando gráfico de Resíduos (SVR)...")
residuos = y_test.values - previsoes
plt.figure(figsize=(15, 7))
plt.scatter(datas_test.values, residuos, alpha=0.7, color='green', marker='.')
plt.axhline(y=0, color='r', linestyle='--')
plt.title(f'Resíduos da Previsão - {modelo.__class__.__name__}')
plt.xlabel('Data')
plt.ylabel('Resíduo (Real - Previsto)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'residuals_plot_{modelo.__class__.__name__}.png')
plt.close()
print(f"DEBUG: Gráfico 'residuals_plot_{modelo.__class__.__name__}.png' salvo.")

# Explicações SHAP para SVM
print("DEBUG: Gerando explicações SHAP para SVR...")
# KernelExplainer é mais lento. Usar uma amostra do X_train para background.
# Se X_train_scaled for muito grande, shap.sample pode demorar.
if X_train_scaled.shape[0] > 100:
    print(f"DEBUG: Criando background para SHAP com 100 amostras de X_train_scaled (tamanho {X_train_scaled.shape[0]})")
    summary_background = shap.sample(X_train_scaled, 100)
elif X_train_scaled.shape[0] > 0:
    print(f"DEBUG: Criando background para SHAP com todas as amostras de X_train_scaled (tamanho {X_train_scaled.shape[0]})")
    summary_background = X_train_scaled
else:
    summary_background = None
    print("ERRO: X_train_scaled está vazio, SHAP não pode ser calculado.")

if summary_background is not None and X_test_scaled.shape[0] > 0:
    try:
        explainer_shap_svm = shap.KernelExplainer(modelo.predict, summary_background)
        # Calcular SHAP values para algumas instâncias de teste (ex: primeiras 50) para economizar tempo
        num_shap_samples = min(50, X_test_scaled.shape[0])
        print(f"DEBUG: Calculando SHAP values para {num_shap_samples} amostras de teste...")
        shap_values_svm = explainer_shap_svm.shap_values(X_test_scaled[:num_shap_samples,:])

        # Plot do SHAP Summary (usando as features names originais)
        plt.figure(figsize=(10,8))
        shap.summary_plot(shap_values_svm, X_test.iloc[:num_shap_samples,:], plot_type="bar", show=False)
        plt.title(f'Importância das Features (SHAP Summary) - {modelo.__class__.__name__}')
        plt.tight_layout()
        plt.savefig(f'shap_summary_{modelo.__class__.__name__}.png')
        plt.close()
        print(f"DEBUG: Gráfico 'shap_summary_{modelo.__class__.__name__}.png' salvo.")
    except Exception as e:
        print(f"ERRO ao gerar SHAP para SVM: {e}")
elif X_test_scaled.shape[0] == 0:
    print("DEBUG: X_test_scaled está vazio. SHAP não será gerado.")

# Explicações LIME para SVM
print("DEBUG: Gerando explicações LIME para SVR (primeira instância de teste)...")
if X_train_scaled.shape[0] > 0 and X_test_scaled.shape[0] > 0:
    try:
        explainer_lime_svm = lime_tabular.LimeTabularExplainer(
            training_data=X_train_scaled, # LIME usa os dados de treino (escalados para SVM)
            feature_names=X_train.columns.tolist(), # Nomes das features originais
            class_names=['Target_Close_D+1'], # Nome da variável alvo
            mode='regression'
        )

        # Função de predição para LIME deve lidar com a normalização.
        # LIME perturba os dados na escala original, então precisamos escalar antes de passar para o modelo SVM.
        def predict_fn_lime_svm(data_lime_original_scale):
            data_lime_scaled = scaler_X.transform(data_lime_original_scale)
            return modelo.predict(data_lime_scaled)

        explicacao_lime_svm = explainer_lime_svm.explain_instance(
            data_row=X_test.iloc[0].values, # Instância original não escalada para LIME
            predict_fn=predict_fn_lime_svm, # Função de predição que escala internamente
            num_features=len(X_test.columns) # Mostrar todas as features
        )
        explicacao_lime_svm.save_to_file(f'lime_explanation_{modelo.__class__.__name__}_instance_0.html')
        print(f"DEBUG: Explicação LIME 'lime_explanation_{modelo.__class__.__name__}_instance_0.html' salva.")
    except Exception as e:
        print(f"ERRO ao gerar LIME para SVM: {e}")
else:
    print("DEBUG: Conjunto de treino ou teste escalado está vazio. LIME não será gerado para SVM.")

print("\nDEBUG: Script SVM.py concluído.")