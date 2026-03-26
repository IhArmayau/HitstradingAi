import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Create dummy data (Replace this with real CSV data later for better accuracy)
# We're creating 1000 samples of "fake" price movements
X_train = np.random.rand(1000, 10, 1) # (samples, time_steps, features)
y_train = np.random.rand(1000, 1)

# 2. Build the LSTM Architecture
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(10, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 3. Train for a few epochs just to initialize the weights
print("Training the QuikPulse AI engine...")
model.fit(X_train, y_train, batch_size=32, epochs=5)

# 4. Save to the correct location
model.save('models/lstm_model.h5')
print("✅ Success! models/lstm_model.h5 is now a real AI model.")
