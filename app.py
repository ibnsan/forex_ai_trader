import MetaTrader5 as mt5
import pandas as pd
import talib
import tensorflow as tf
from sklearn.model_selection import train_test_split


if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

symbol = "EURUSD"
rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M15, mt5.datetime(), 1000)
mt5.shutdown()

df = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])

df['rsi'] = talib.RSI(df['close'], timeperiod=14)
macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['macd'] = macd
df['macd_signal'] = macdsignal
df['macd_hist'] = macdhist

upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
df['bb_upper'] = upper
df['bb_middle'] = middle
df['bb_lower'] = lower

df['hour'] = df['time'].dt.hour
df['minute'] = df['time'].dt.minute
df['dayofweek'] = df['time'].dt.dayofweek

lag_features = ['open', 'high', 'low', 'close', 'rsi', 'macd', 'macd_signal', 'macd_hist']
for feature in lag_features:
    df[f'{feature}_lag1'] = df[feature].shift(1)

df['target'] = df['close'].shift(-1) - df['close']

# Removing rows with missing values
df.dropna(inplace=True)

X = df.drop(columns=['time', 'target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_dim=X_train.shape[1]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

