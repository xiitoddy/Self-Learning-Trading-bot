# Self-Learning-Trading-bot
The code provided is a trading algorithm that uses various technical indicators to analyze historical stock data and make buy/sell decisions. It imports necessary libraries such as Pandas, NumPy, TensorFlow, and Gym for data manipulation, machine learning, and reinforcement learning, respectively. The Alpaca API is used to fetch historical stock data.

The get_data function fetches daily stock data and calculates technical indicators, including Simple Moving Average (SMA), Exponential Moving Average (EMA), Bollinger Bands, Relative Strength Index (RSI), Stochastic Oscillator, Moving Average Convergence Divergence (MACD), Average Directional Index (ADX), and Average True Range (ATR). Additionally, it computes Fibonacci retracement and extension levels, and pivot points.

The preprocess_data function removes the 'returns' column, handles missing or invalid values, and converts the data into NumPy arrays for further processing.

The define_model function creates a simple feedforward neural network using TensorFlow and Keras, with two hidden layers having 32 and 16 neurons, respectively.

The create_env function sets up a custom trading environment using the Gym library. The environment's state includes stock data along with account information (shares, balance, and equity). The action space consists of three possible actions: buy, sell, and hold.

The train_model function trains the model using reinforcement learning with a Q-learning algorithm, allowing it to explore and exploit the environment, with a decaying epsilon for the epsilon-greedy strategy.

The moving_average_strategy function is a simple trading strategy that compares the stock's closing price to its simple moving average. If the closing price is above the SMA by a specified threshold, it signals a buy. If the closing price is below the SMA by a specified threshold, it signals a sell.

The main function serves as the entry point of the code. It fetches the historical stock data for the specified symbol, preprocesses the data, and creates a trading environment using the moving average strategy. The environment is then used to train the model.

In summary, the code is an algorithmic trading system that utilizes machine learning and reinforcement learning techniques to make buy/sell decisions based on historical stock data and technical indicators. It also provides an example of how to create a custom trading environment using the Gym library.

