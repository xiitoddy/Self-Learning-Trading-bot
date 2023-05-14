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
import pandas as pd
from ta import add_all_ta_features
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.momentum import StochasticOscillator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange
from ta.trend import MACD
from collections import deque
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import keras.backend as K
K.clear_session()

ALPACA_API_KEY = 'PK59CQ7FVC00MI7ZQ2D4'
ALPACA_SECRET_KEY = 'MGz923hgdl6qC12Rtk1OjEe1Zk87uctn3cGlmTzB'
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

def get_data(symbol, start, end):
    data = api.get_bars(
        symbol,
        '15Min',        
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

    try:
        retracement_levels = calculate_fibonacci_retracement_levels(data)
        extension_levels = calculate_fibonacci_extension_levels(data)
        pivot_points = calculate_pivot_points(data)
    except Exception as e:
        print(f"Error calculating Fibonacci levels or pivot points: {e}")

    print("Data:")
    print(data.head())

    return data

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

    print("Missing values in the data:")
    print(data.isna().sum())

    print("Infinite values in the data:")
    print(np.isinf(data).sum())

    data = data.fillna(method='ffill')  # Forward-fill missing values
    data = data.fillna(method='bfill')  # Back-fill any remaining missing values

    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    x = data.values
    y = np.where(x[:, -1] > 0, 1, 0)
    data = data.fillna(method='ffill')

    print("\nPreprocessed data:")
    print("X:", x[:5])
    print("Y:", y[:5])

    return x, y

def define_model(input_shape, output_shape):  
    model = keras.Sequential([  
    layers.LSTM(32, activation='relu', input_shape=(60, 27)),  
    layers.Dense(16, activation='relu'),  
    layers.Dense(output_shape, activation='linear')  
    ])  

    model.compile(loss='mse', optimizer='adam')  
    return model


def create_env(data, strategy):
    class TradingEnvironment(gym.Env):
        def __init__(self, data):
            self.data = data
            self.action_space = gym.spaces.Discrete(3)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(data.shape[1] + 3,), dtype=np.float32)
            self.trades_log = []
            self.stop_loss = None
            self.last_action = None
            self.reset()

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
            return observation

        def step(self, action):
          
            if self.done:
                raise Exception("Trading environment is done, please reset.")

            if hasattr(self, 'waiting_for_sell_prediction') and self.waiting_for_sell_prediction and action != 1:
                return self._get_observation(), 0, self.done, {}  # Skip the step if waiting for a sell prediction

            close_price = self.data.iloc[self.index]['close']
            reward = 9
            done = False
            info = {}
            TRANSACTION_COST = 0.0005
            long_position_penalty = 0.009
            action_penalty = 0.0010
            MAX_LOSS_PERCENT = 0.02  # Maximum 2% loss per trade
            RISK_PERCENTAGE = 0.1  # Use only 10% of the balance for each trade           

            if action == 0:  # Buy
                shares_to_buy = (self.balance * RISK_PERCENTAGE) / close_price
                risk_per_share = close_price * MAX_LOSS_PERCENT
                stop_loss_price = close_price - risk_per_share
                shares_to_buy = min(shares_to_buy, self.balance // risk_per_share)

                if shares_to_buy > 0:
                    self.shares += shares_to_buy
                    self.balance -= shares_to_buy * close_price * (1 - TRANSACTION_COST)
                    self.positions.append((self.index, shares_to_buy))
                    self.stop_loss = stop_loss_price
                    info['action'] = 'buy'
                    print(f"Buy: {shares_to_buy} shares at {close_price}")
                    print(f"Updated Balance: {self.balance}")
                    trade = {
                        "action": "buy",
                        "shares": shares_to_buy,
                        "price": close_price,
                        "timestamp": self.data.index[self.index]
                    }
                    self.trades_log.append(trade)
                    self.waiting_for_sell_prediction = True

                else:
                  reward -= action_penalty
            
            elif action == 1:  # Sell
                if len(self.positions) == 0 or close_price <= self.stop_loss:
                    self.done = True
                    return self._get_observation(), reward, self.done, info

                position = self.positions.pop(0)
                shares_to_sell = position[1]
                self.shares -= shares_to_sell
                self.balance += shares_to_sell * close_price * (1 - TRANSACTION_COST)
                profit = (close_price - self.data.iloc[position[0]]['close']) * shares_to_sell
                self.profits.append(profit)
                reward = profit  # Modify this line to use profit as a reward
                info['action'] = 'sell'
                info['profit'] = profit
                print(f"Sell: {shares_to_sell} shares at {close_price}, Profit: {profit}")
                print(f"Updated Balance: {self.balance}")
                # Log the trade
                trade = {
                    "action": "sell",
                    "shares": shares_to_sell,
                    "price": close_price,
                    "timestamp": self.data.index[self.index],
                    "profit": profit
                }
                self.trades_log.append(trade)
                self.waiting_for_sell_prediction = False
            else:
                reward -= action_penalty
                
            if self.index == len(self.data) - 1:
                self.done = True

            for position in self.positions:
              if self.index - position[0] > 5:
                reward -= long_position_penalty

            returns = np.array(self.profits)
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0

            # Calculate the profit factor
            gross_profit = np.sum(returns[returns > 0])
            gross_loss = np.abs(np.sum(returns[returns < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0

            self.index += 1
            observation = self._get_observation()

            self.equity = self.balance + self.shares * close_price
            reward = (self.equity - self.baseline_profit) / self.baseline_profit + sharpe_ratio + profit_factor

            return observation, reward, self.done, info

        def _get_observation(self):
          if self.index >= len(self.data):
            self.done = True
            return np.zeros((self.observation_space.shape[0],))
        
          state = self.data.iloc[self.index].values
          if state.ndim == 0:
            state = np.array([state])

          state = np.concatenate(([self.shares, self.balance, self.equity], state))
          return state

       
        def save_trades(self, filepath):
          trades_df = pd.DataFrame(self.trades_log)
          trades_df['balance'] = self.balance
          trades_df.to_csv(filepath, index=False)

    env = TradingEnvironment(data)
    action_space = gym.spaces.Discrete(3)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1] + 2,), dtype=np.float32)
    env = TradingEnvironment(data)
    env.strategy = strategy
    
    return env

from collections import deque

def train_model(x, model, episodes, batch_size, env):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    action_space = np.arange(action_size)
    np.random.seed(123)
    total_size = x.shape[0] * x.shape[1]
    truncated_size = (total_size // (60 * state_size)) * (60 * state_size)
    x = x.flatten()[:truncated_size]
    x = np.reshape(x, (-1, 60, state_size))
    
    state_buffer = deque(maxlen=60)  # buffer for the last 60 states

    for e in range(episodes):
        # reset the environment
        state = env.reset()
        state_buffer.append(state)  # add the initial state to the buffer
        done = False
        i = 0
        while not done:
            if len(state_buffer) < 60:
                # if the buffer is not full yet, choose a random action
                action = np.random.randint(0, action_size)
            else:
                # if the buffer is full, reshape it to match the model's input shape and make a prediction
                state_input = np.reshape(np.array(state_buffer), (-1, 60, state_size))
                action = np.argmax(model.predict(state_input)[0])
            next_state, reward, done, _ = env.step(action)
            state_buffer.append(next_state)  # add the new state to the buffer
            
            next_state = np.concatenate(([env.shares, env.balance, env.equity], next_state)) # add the account information to the next_state
            
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, episodes, i, 0.01))
                break
            if len(state_buffer) == 60:
                model.fit(x, batch_size=batch_size, verbose=0)
            i += 1
    return model

def moving_average_strategy(data, buy_threshold=0.02, sell_threshold=0.01):
    action = 0
    if data['close'] > data['sma'] * (1 + buy_threshold):
        action = 0  # Buy
    elif data['close'] < data['sma'] * (1 - sell_threshold):
        action = 1  # Sell
    else:
        action = 2  # Hold
    return action

def save_trades_to_csv(trades, filepath):
    with open(filepath, mode='w', newline='') as file:
        fieldnames = ["action", "shares", "price", "timestamp", "profit"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            writer.writerow(trade)

def main():
    symbol = 'MSFT'
    '15Min',
    start_date = '2019-01-01'
    end_date = '2022-12-30'
    data = get_data(symbol, start_date, end_date)
    x, y = preprocess_data(data)

    data = get_data('MSFT', '2019-01-01', '2022-12-30')
    
    print("Raw data:")
    print(data.head())
    
    print("\nPreprocessed data:")
    print("X:", x[:5])
    print("Y:", y[:5])

    env = create_env(data, strategy=moving_average_strategy)
    
    print("Original x shape:", x.shape)
    print("Model's input shape:", env.observation_space.shape)
    
    model = define_model(env.observation_space.shape, env.action_space.n)  # Corrected line

    episodes = 20
    batch_size = 32

    model = train_model(x, model, episodes, batch_size, env)


    model.save('trained_model.h5')
    env.save_trades("trades.csv")

    test_data = get_data('MSFT', '2019-01-01', '2022-12-30')
    test_env = create_env(test_data, moving_average_strategy)
    state = test_env.reset()
    done = False
    while not done:
        state = np.concatenate(([test_env.shares, test_env.balance, test_env.equity], state))
        state = state.reshape((1, -1))
        action = np.argmax(model.predict(state)[0])
        state, reward, done, info = test_env.step(action)

    print("Test results:")
    print(f"Final equity: {test_env.equity}")
    print(f"Total profit: {sum(test_env.profits)}")

if __name__ == '__main__':
    main()
