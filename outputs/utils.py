import pandas as pd
import datetime as dt

def criar_features_defasadas(df, nome_coluna_base, num_lags, prefixo_coluna_lag='Lag_'):
    """
    Cria features defasadas (lags) para uma coluna específica.
    Exemplo de nome de coluna gerada: Lag_1_Close, Lag_2_Volume.
    """
    print(f"DEBUG: Criando lags para {nome_coluna_base} com prefixo {prefixo_coluna_lag} e {num_lags} lags.")
    for i in range(1, num_lags + 1):
        df[f'{prefixo_coluna_lag}{i}_{nome_coluna_base}'] = df[nome_coluna_base].shift(i)
    return df

def criar_features_data(df, nome_coluna_data):
    """
    Cria features de data (Ano, Mês, Dia, Dia da Semana) a partir de uma coluna de data.
    """
    print(f"DEBUG: Criando features de data para {nome_coluna_data}")
    if nome_coluna_data not in df.columns:
        print(f"ERRO: Coluna de data '{nome_coluna_data}' não encontrada no DataFrame.")
        return df
    
    # Garante que a coluna de data está no formato datetime
    df[nome_coluna_data] = pd.to_datetime(df[nome_coluna_data])
    
    df['Year'] = df[nome_coluna_data].dt.year
    df['Month'] = df[nome_coluna_data].dt.month
    df['Day'] = df[nome_coluna_data].dt.day
    df['DayOfWeek'] = df[nome_coluna_data].dt.dayofweek # Segunda=0, Domingo=6
    return df 