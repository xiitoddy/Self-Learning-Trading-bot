from cmath import e
import pandas as pd
import numpy as np
import gym
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import alpaca_trade_api as tradeapi
from alpaca_trade_api import TimeFrame
import pyti
import ta
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.momentum import StochasticOscillator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange
from ta.trend import MACD
from collections import deque

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import keras.backend as K
K.clear_session()

ALPACA_API_KEY = 'PK265DGP9ZU4IO3H9JRI'
ALPACA_SECRET_KEY = 'ttY8GfYCcBDjNhi7r3uqoIUYNqxdLaDfO3uErvz9'
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

def get_data(symbol, start, end):
    data = api.get_bars(
        symbol,
        TimeFrame.Day,
        start=start,
        end=end,
    ).df
    data.dropna(inplace=True)
    data['returns'] = data['close'].pct_change()

    # Technical Indicators
    try:
        sma = SMAIndicator(data['close'], window=14)
        data['sma'] = sma.sma_indicator()
    except Exception as e:
        print(f"Error calculating SMA: {e}")

    try:
        ema = EMAIndicator(data['close'], window=14)
        data['ema'] = ema.ema_indicator()
    except Exception as e:
        print(f"Error calculating EMA: {e}")

    try:
        bb = BollingerBands(data['close'], window=14, fillna=0)
        data['upper_bb'] = bb.bollinger_hband()
        data['middle_bb'] = bb.bollinger_mavg()
        data['lower_bb'] = bb.bollinger_lband()
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")

    try:
        rsi = RSIIndicator(data['close'], window=14)
        data['rsi'] = rsi.rsi()
    except Exception as e:
        print(f"Error calculating RSI: {e}")

    try:
        stoch = StochasticOscillator(data['high'], data['low'], data['close'], window=14, smooth_window=3)
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()
    except Exception as e:
        print(f"Error calculating Stochastic Oscillator: {e}")

    try:
        macd = MACD(data['close'], window_fast=12, window_slow=26, window_sign=9)
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_histogram'] = macd.macd_diff()
    except Exception as e:
        print(f"Error calculating MACD: {e}")

    try:
        adx = ADXIndicator(data['high'], data['low'], data['close'], window=14)
        data['adx'] = adx.adx()
    except Exception as e:
        print(f"Error calculating ADX: {e}")

    try:
        atr = AverageTrueRange(data['high'], data['low'], data['close'], window=14)
        data['atr'] = atr.average_true_range()
    except Exception as e:
        print(f"Error calculating ATR: {e}")

    # Calculate Fibonacci retracement levels, extensions, and pivot points (not added to the data DataFrame)
    try:
        retracement_levels = calculate_fibonacci_retracement_levels(data)
        extension_levels = calculate_fibonacci_extension_levels(data)
        pivot_points = calculate_pivot_points(data)
    except Exception as e:
        print(f"Error calculating Fibonacci levels or pivot points: {e}")

    # Add more technical indicators if needed...

    print("Data:")
    print(data.head())

    return data

# technical indicators helper functions

def calculate_fibonacci_retracement_levels(data):
    max_price = data['high'].max()
    min_price = data['low'].min()
    diff = max_price - min_price

    levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    retracement_levels = [(max_price - l * diff) for l in levels]

    return retracement_levels


def calculate_fibonacci_extension_levels(data):
    max_price = data['high'].max()
    min_price = data['low'].min()
    diff = max_price - min_price

    levels = [0.0, 1.0, 1.618, 2.0, 2.618, 3.0]
    extension_levels = [(max_price + l * diff) for l in levels]

    return extension_levels  # Add this line

def calculate_pivot_points(data):
    pivot_point = (data['high'].iloc[-1] + data['low'].iloc[-1] + data['close'].iloc[-1]) / 3
    support1 = 2 * pivot_point - data['high'].iloc[-1]
    resistance1 = 2 * pivot_point - data['low'].iloc[-1]
    support2 = pivot_point - (data['high'].iloc[-1] - data['low'].iloc[-1])
    resistance2 = pivot_point + (data['high'].iloc[-1] - data['low'].iloc[-1])
    support3 = pivot_point - 2 * (data['high'].iloc[-1] - data['low'].iloc[-1])
    resistance3 = pivot_point + 2 * (data['high'].iloc[-1] - data['low'].iloc[-1])

    pivot_points = [pivot_point, support1, resistance1, support2, resistance2, support3, resistance3]

    return pivot_points

def preprocess_data(data):
    # Drop the 'returns' column
    data = data.drop(['returns'], axis=1)

    # Check for missing or invalid values
    print("Missing values in the data:")
    print(data.isna().sum())

    print("Infinite values in the data:")
    print(np.isinf(data).sum())

    # Handle missing or invalid values
    data = data.fillna(method='ffill')  # Forward-fill missing values
    data = data.fillna(method='bfill')  # Back-fill any remaining missing values

    # Drop last 4 columns and take the first 20 columns
    data = data.iloc[:, :-4]

    # Convert to NumPy arrays
    x = data.values
    y = np.where(x[:, -1] > 0, 1, 0)
    data = data.fillna(method='ffill')

    print("\nPreprocessed data:")
    print("X:", x[:5])
    print("Y:", y[:5])

    return x, y

def define_model(input_shape, output_shape):
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(27,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(output_shape, activation='linear')
    ])

    model.compile(loss='mse', optimizer='adam')
    return model


def create_env(data, strategy):
    class TradingEnvironment(gym.Env):
        def __init__(self, data):
            print("Creating TradingEnvironment...")
            self.data = data
            self.action_space = gym.spaces.Discrete(3)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(data.shape[1] + 3,), dtype=np.float32)
            self.reset()

            print('State shape:', self.data.iloc[0].values.shape)
            print('Observation space:', self.observation_space)
            print('Action space:', self.action_space)


        def reset(self):
            self.index = 0
            self.positions = []
            self.profits = []
            self.baseline_profit = self.data.iloc[0]['close']
            self.balance = 1e6
            self.shares = 0
            self.equity = self.balance + self.shares * self.data.iloc[self.index]['close']
            self.done = False
            observation = self._get_observation()
            print("Observation shape:", observation.shape)
            return observation

        def step(self, action):
            if self.done:
                raise Exception("Trading environment is done, please reset.")

            close_price = self.data.iloc[self.index]['close']
            reward = 0
            done = False
            info = {}

            if action == 0:  # Buy
                shares_to_buy = self.balance / close_price
                self.shares += shares_to_buy
                self.balance -= shares_to_buy * close_price
                self.positions.append((self.index, shares_to_buy))
                info['action'] = 'buy'
            elif action == 1:  # Sell
                if len(self.positions) == 0:
                    self.done = True
                    return self._get_observation(), reward, self.done, info

                position = self.positions.pop(0)
                shares_to_sell = position[1]
                self.shares -= shares_to_sell
                self.balance += shares_to_sell * close_price
                profit = (close_price - self.data.iloc[position[0]]['close']) * shares_to_sell
                self.profits.append(profit)
                info['action'] = 'sell'
                info['profit'] = profit

            if self.index == len(self.data) - 1:
                self.done = True

            self.equity = self.balance + self.shares * close_price
            reward = (self.equity - self.baseline_profit) / self.baseline_profit

            self.index += 1
            observation = self._get_observation()
            return observation, reward, self.done, info

        def _get_observation(self):
          print(f"Index: {self.index}, Data shape: {self.data.shape}")
          if self.index >= len(self.data):
            self.done = True
            return np.zeros((self.observation_space.shape[0],))
        
          state = self.data.iloc[self.index].values
          if state.ndim == 0:
            state = np.array([state])

          state = np.concatenate(([self.shares, self.balance, self.equity], state))
          return state


    # The following block should be indented at the same level as the class definition
    env = TradingEnvironment(data)
    print('State shape:', env.data.iloc[0].values.shape)
    print('Observation space:', env.observation_space)
    print('Action space:', env.action_space)
    action_space = gym.spaces.Discrete(3)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1] + 2,), dtype=np.float32)
    env = TradingEnvironment(data)
    env.strategy = strategy
    
    return env


def train_model(x, model, episodes, batch_size, env):
    action_size = env.action_space.n
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.95

    memory = deque(maxlen=2000)

    for episode in range(episodes):
        state = env.reset()
        state_size = x.shape[1] + 4
        state = np.concatenate(([env.shares, env.balance, env.equity], state)) # add the account information to the state
        state = state.reshape((1, -1)) # add an extra dimension to the state variable
        done = False
        while not done:
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, action_size)
            else:
                action = np.argmax(model.predict(state)[0])
            next_state, reward, done, _ = env.step(action)
            next_state = np.concatenate(([env.shares, env.balance, env.equity], next_state)) # add the account information to the next_state
            next_state = next_state.reshape((1, -1)) # add an extra dimension to the next_state variable
            memory.append((state, action, reward, next_state, done))
            state = next_state

        if len(memory) >= batch_size:
            minibatch = random.sample(memory, batch_size)
            X_state = []
            X_target = []
            for state, action, reward, next_state, done in minibatch:
                if done:
                    target = reward
                else:
                    target = reward + gamma * np.amax(model.predict(next_state)[0])
                target_f = model.predict(state)
                target_f[0][action] = target
                X_state.append(state.reshape((1, -1)))
                X_target.append(target_f.reshape((1, -1)))
            X_state = np.concatenate(X_state, axis=0)
            X_target = np.concatenate(X_target, axis=0)
            model.fit(X_state, X_target, epochs=1, verbose=0)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    return model

def moving_average_strategy(data, buy_threshold=0.02, sell_threshold=0.02):
    action = 0
    if data['close'] > data['sma'] * (1 + buy_threshold):
        action = 0  # Buy
    elif data['close'] < data['sma'] * (1 - sell_threshold):
        action = 1  # Sell
    return action


def main():
    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2022-12-30'
    data = get_data(symbol, start_date, end_date)
    x, y = preprocess_data(data)

    data = get_data('AAPL', '2021-01-01', '2021-12-31')
    
    print("Raw data:")
    print(data.head())
    
    print("\nPreprocessed data:")
    print("X:", x[:5])
    print("Y:", y[:5])

    # Print the shapes of the model input and the data
    print("Model input shape:", define_model((x.shape[1],), 3).input_shape)
    print("Data shape:", x.shape)

    # Print the data
    print("Data:")
    print(data.head())




    strategy = moving_average_strategy  # Define your strategy here
    env = create_env(data, strategy)

    print("Initial observation shape:", env.reset().shape)
    print('State shape:', env.data.iloc[0].values.shape)
    print('Observation space:', env.observation_space)
    print('Action space:', env.action_space)
    
    print(f"Observation space shape: {env.observation_space.shape}")

    model = define_model((x.shape[1],), 3)
    episodes = 5
    batch_size = 64
    model = train_model(x, model, episodes, batch_size, env)

    print("Training complete.")
    model.save('trained_model.h5')
    print("Model saved.")

    for layer in model.layers:
      print(layer.name)
      print("Weights shape:", layer.get_weights()[0].shape)
      print("Biases shape:", layer.get_weights()[1].shape)


    # Make predictions using the trained model and the data
    print("Input data shape:", x.shape)
    predictions = model.predict(x)
    print("Predictions shape:", predictions.shape)

if __name__ == '__main__':
    main()
