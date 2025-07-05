import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular
from outputs.utils import criar_features_defasadas, criar_features_data

print("DEBUG: Módulos importados")

# Funções criar_features_defasadas e criar_features_data foram movidas para utils.py

# Carregar os dados
print("DEBUG: Carregando dados...")
try:
    # Ajuste o caminho se o seu arquivo CSV estiver em outro local
    df_original = pd.read_csv("BTC-USD.csv")
    print("DEBUG: Dados crus carregados:")
    print(df_original.head())
except FileNotFoundError:
    print("ERRO: Arquivo BTC-USD.csv não encontrado. Certifique-se de que está no mesmo diretório do script.")
    exit()

df = df_original.copy()

# Convertendo a coluna 'Date' para datetime, se ainda não estiver
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
else:
    print("ERRO: Coluna 'Date' não encontrada no CSV.")
    exit()

# Engenharia de Features
print("DEBUG: Iniciando engenharia de features...")
N_LAGS = 5 # Número de lags a serem criados para cada feature
COLUNAS_PARA_LAGS = ['Open', 'High', 'Low', 'Close', 'Volume'] # Features do dia D

# Criar o alvo (Close do dia seguinte)
df['Target_Close_D+1'] = df['Close'].shift(-1)

# Criar lags para as colunas especificadas
for coluna in COLUNAS_PARA_LAGS:
    df = criar_features_defasadas(df, coluna, N_LAGS, prefixo_coluna_lag=f'Lag{coluna}_')

# Criar features de data a partir da coluna 'Date'
df = criar_features_data(df, 'Date')

# Remover linhas com NaN resultantes da criação de lags e do target_shift
print(f"DEBUG: Linhas antes do dropna: {len(df)}")
df.dropna(inplace=True)
print(f"DEBUG: Linhas após o dropna: {len(df)}")

if df.empty:
    print("ERRO: DataFrame vazio após dropna. Verifique a criação de lags e do target.")
    exit()

# Seleção de Features (X) e Alvo (y)
# Features são todas as colunas do dia D (Open, High, Low, Close, Volume),
# todos os lags criados, e as features de data.
# Excluímos 'Date' (já processada), 'Adj Close' (se existir, para evitar leakage e porque já temos Close)
# e o próprio 'Target_Close_D+1'.

# Lista inicial de features potenciais
features_potenciais = [
    'Open', 'High', 'Low', 'Close', 'Volume', # Dados do dia D
    'Year', 'Month', 'Day', 'DayOfWeek' # Features de data do dia D
]
# Adicionar colunas de lag à lista de features
for coluna_base in COLUNAS_PARA_LAGS:
    for i in range(1, N_LAGS + 1):
        features_potenciais.append(f'Lag{coluna_base}_{i}_{coluna_base}') # Corrigido para o novo padrão de nome

# Garantir que todas as features potenciais existem no DataFrame após o dropna
features_finais = [f for f in features_potenciais if f in df.columns]

# Remover colunas que não devem ser features (como 'Adj Close' se estiver presente, ou o target)
features_a_remover = ['Date', 'Adj Close', 'Target_Close_D+1'] # 'Adj Close' pode ou não estar, mas é bom garantir
features_finais = [f for f in features_finais if f not in features_a_remover and f in df.columns]

print(f"DEBUG: Features finais selecionadas para o modelo: {features_finais}")

X = df[features_finais]
y = df['Target_Close_D+1']

# Guardar as datas para os plots, antes da divisão e após o dropna
datas_plot = df['Date']

# Divisão dos dados em treino e teste (CRONOLÓGICO)
# Usar train_test_split com shuffle=False para manter a ordem temporal
TEST_SIZE = 0.2
X_train, X_test, y_train, y_test, datas_train, datas_test = train_test_split(
    X, y, datas_plot, test_size=TEST_SIZE, shuffle=False
)

print(f"DEBUG: Tamanho do conjunto de treino: {X_train.shape[0]}")
print(f"DEBUG: Tamanho do conjunto de teste: {X_test.shape[0]}")

if X_train.empty or X_test.empty:
    print("ERRO: Conjunto de treino ou teste vazio. Verifique o tamanho dos dados e o TEST_SIZE.")
    exit()

# Treinamento do Modelo (Random Forest como exemplo)
print("DEBUG: Treinando o modelo RandomForestRegressor...")
modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)
print("DEBUG: Modelo treinado.")

# Previsões
print("DEBUG: Fazendo previsões...")
previsoes = modelo.predict(X_test)

# Avaliação do Modelo
print("DEBUG: Avaliando o modelo...")
mse = mean_squared_error(y_test, previsoes)
mae = mean_absolute_error(y_test, previsoes)
r2 = r2_score(y_test, previsoes)

print(f"\n--- Resultados do Modelo ({modelo.__class__.__name__}) ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Plot: Previsões vs Valores Reais
print("DEBUG: Gerando gráfico Previsões vs Reais...")
plt.figure(figsize=(15, 7))
plt.plot(datas_test, y_test.values, label='Valores Reais', color='blue', marker='o', linestyle='-')
plt.plot(datas_test, previsoes, label='Previsões', color='red', marker='x', linestyle='--')
plt.title(f'Previsão de Preço (Close D+1) - {modelo.__class__.__name__}')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show() # Comentado para execução em lote
plt.savefig(f'prediction_vs_actual_{modelo.__class__.__name__}.png')
plt.close()
print(f"DEBUG: Gráfico 'prediction_vs_actual_{modelo.__class__.__name__}.png' salvo.")

# Plot: Resíduos
print("DEBUG: Gerando gráfico de Resíduos...")
residuos = y_test - previsoes
plt.figure(figsize=(15, 7))
plt.scatter(datas_test, residuos, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.title(f'Resíduos da Previsão - {modelo.__class__.__name__}')
plt.xlabel('Data')
plt.ylabel('Resíduo (Real - Previsto)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show() # Comentado para execução em lote
plt.savefig(f'residuals_plot_{modelo.__class__.__name__}.png')
plt.close()
print(f"DEBUG: Gráfico 'residuals_plot_{modelo.__class__.__name__}.png' salvo.")


# Explicações SHAP
print("DEBUG: Gerando explicações SHAP...")
# Para RandomForest, TreeExplainer é mais eficiente
explainer_shap = shap.TreeExplainer(modelo)
shap_values = explainer_shap.shap_values(X_test)

# Plot do SHAP Summary
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title(f'Importância das Features (SHAP Summary) - {modelo.__class__.__name__}')
plt.tight_layout()
# plt.show() # Comentado para execução em lote
plt.savefig(f'shap_summary_{modelo.__class__.__name__}.png')
plt.close()
print(f"DEBUG: Gráfico 'shap_summary_{modelo.__class__.__name__}.png' salvo.")

# Explicações LIME para uma instância específica
print("DEBUG: Gerando explicações LIME para a primeira instância de teste...")
# LIME espera um preditor que retorne probabilidades para classificação, 
# ou valores numéricos para regressão. predict() do RandomForest já faz isso.
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Target_Close_D+1'], # Nome da variável alvo
    mode='regression'
)

if not X_test.empty:
    explicacao_lime = explainer_lime.explain_instance(
        data_row=X_test.iloc[0].values, # Primeira instância do conjunto de teste
        predict_fn=modelo.predict,      # Função de predição do modelo treinado
        num_features=len(X_test.columns) # Mostrar todas as features
    )
    # Salvar a explicação LIME em HTML
    try:
        explicacao_lime.save_to_file(f'lime_explanation_{modelo.__class__.__name__}_instance_0.html')
        print(f"DEBUG: Explicação LIME 'lime_explanation_{modelo.__class__.__name__}_instance_0.html' salva.")
    except Exception as e:
        print(f"ERRO ao salvar LIME HTML: {e}")
else:
    print("DEBUG: X_test está vazio, LIME não será gerado.")

print("\nDEBUG: Script Arvores_de_Decisao.py concluído.")