import pandas as pd
import datetime as dt

# Carregar o arquivo CSV
nome_arquivo = 'BTC-USD.csv'

nome_arquivo_sem_sufixo = nome_arquivo.split(".")[0]
df = pd.read_csv(nome_arquivo)

# Exibir as primeiras linhas do DataFrame original
print("DataFrame original:")
print(df.head())


df['DateTime'] = pd.to_datetime(df['Date'])
df['DateTime'] = df['DateTime'].map(dt.datetime.toordinal)

# Remover algumas colunas
colunas_para_remover = ['Date', 'Open','High','Low','Adj Close','Volume']  # Substitua 'coluna1', 'coluna2' pelos nomes das colunas que deseja remover
df = df.drop(columns=colunas_para_remover)

# Exibir as primeiras linhas do DataFrame após a remoção das colunas
print("\nDataFrame após a remoção das colunas:")
print(df.head())

# Salvar o DataFrame em um novo arquivo CSV
novo_nome_arquivo = nome_arquivo_sem_sufixo + "_PreProcessado.csv"  # Nome do novo arquivo CSV
df.to_csv(novo_nome_arquivo, index=False)  # index=False para não incluir índices no arquivo CSV

print(f"\nDataFrame salvo com sucesso como '{novo_nome_arquivo}'.")