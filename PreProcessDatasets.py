import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

# Carregar o arquivo CSV
nome_arquivo = 'datasets/coin_Bitcoin.csv'

nome_arquivo_sem_sufixo = nome_arquivo.split(".")[0]
df = pd.read_csv(nome_arquivo)

# Exibir as primeiras linhas do DataFrame original
print("DataFrame original:")
print(df.head())

# Converter a coluna 'Date' para o formato ordinal
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].map(dt.datetime.toordinal)

# Remover algumas colunas
colunas_para_remover = ['SNo', 'Name','Symbol', 'Marketcap']
df = df.drop(columns=colunas_para_remover)

# Normalizar colunas específicas usando MinMaxScaler
# colunas_para_normalizar = [ 'Close', 'Volume', 'High', 'Low', 'Open']
# scaler = MinMaxScaler()
# df[colunas_para_normalizar] = scaler.fit_transform(df[colunas_para_normalizar])

# Exibir as primeiras linhas do DataFrame após a remoção das colunas e normalização
print("\nDataFrame após a remoção das colunas e normalização:")
print(df)

# Salvar o DataFrame em um novo arquivo CSV
novo_nome_arquivo = nome_arquivo_sem_sufixo + "_PreProcessado.csv"  # Nome do novo arquivo CSV
df.to_csv(novo_nome_arquivo, index=False)  # index=False para não incluir índices no arquivo CSV

print(f"\nDataFrame salvo com sucesso como '{novo_nome_arquivo}'.")