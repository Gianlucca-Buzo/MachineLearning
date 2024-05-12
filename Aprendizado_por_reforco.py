import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces


if __name__ == '__main__':
    # Carregar dados
    df_btc = pd.read_csv('BTC-USD_PreProcessado.csv')

    # Definir ambiente Gym
    class BitcoinTradingGym(gym.Env):
        def __init__(self, df):
            super(BitcoinTradingGym, self).__init__()
            self.df = df
            self.max_steps = len(df) - 1  # Ajuste para garantir que o índice máximo seja o último índice válido
            self.reset()

            # Definir espaço de ação
            self.action_space = spaces.Discrete(3)  # Ajuste conforme necessário para o seu problema

            # Definir espaço de observação
            self.observation_space = spaces.Box(low=0, high=1, shape=(len(df.columns),), dtype=np.float32)

        def reset(self):
            self.current_step = 0
            return self.df.iloc[self.current_step, :]

        def reward_function(self, predicted_value, actual_value, action):
            # Política de recompensa simples: recompensa positiva se a previsão estiver correta, negativa caso contrário
            if action == 0:  # Exemplo: prever que o próximo valor será menor
                if predicted_value < actual_value:
                    return 1  # Recompensa positiva
                else:
                    return -1  # Recompensa negativa
            else:  # Exemplo: prever que o próximo valor será maior
                if predicted_value > actual_value:
                    return 1  # Recompensa positiva
                else:
                    return -1  # Recompensa negativa

        def step(self, action):
            self.current_step += 1
            if self.current_step >= self.max_steps:
                done = True
                obs = self.df.iloc[self.current_step - 1, :]  # Usar o último estado
            else:
                done = False
                obs = self.df.iloc[self.current_step, :]

            predicted_value = self.df.loc[self.df.index[self.current_step - 1], 'Close']
            actual_value = self.df.loc[self.df.index[self.current_step], 'Close']
            reward = self.reward_function(predicted_value, actual_value, action)

            info = {}  # Informações adicionais, se necessário

            return obs, reward, done, info

    # Criar ambiente Gym
    env = BitcoinTradingGym(df_btc)

    # Envoltório de vetor para o ambiente
    env = DummyVecEnv([lambda: env])

    # Criar modelo PPO
    model = PPO("MlpPolicy", env, verbose=1)

    # Treinar o modelo
    model.learn(total_timesteps=10000)

    # Testar o modelo
    obs = env.reset()
    rewards_interp = []  # Lista para armazenar as recompensas interpoladas
    for _ in range(len(df_btc) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        rewards_interp.append(rewards[0])  # Apenas o primeiro valor de rewards é usado
        if done:
            break

    # Interpolar os valores de rewards para corresponder às datas
    rewards_interp = np.interp(df_btc.index, np.arange(len(rewards_interp)), rewards_interp)

    # Visualizar resultados
    plt.plot(df_btc['DateTime'], df_btc['Close'], label='Preço Real')
    plt.plot(df_btc['DateTime'], rewards_interp, label='Preço Previsto')
    plt.xlabel('Data')
    plt.ylabel('Preço de Fechamento')
    plt.title('Preço Real vs. Preço Previsto')
    plt.legend()
    plt.show()
