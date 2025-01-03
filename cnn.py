import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

data = pd.read_csv("soybean_samples.csv")

X = data.iloc[:, 3:]
y = data['yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

model = Sequential()
model.add(Conv1D(filters=128, kernel_size=4, activation='relu', input_shape=(X_train_scaled.shape[1], 1), kernel_regularizer=l2(0.0001)))
model.add(Dropout(0.15))
model.add(BatchNormalization())
model.add(Conv1D(filters=128, kernel_size=4, activation='relu', kernel_regularizer=l2(0.0001)))
model.add(Dropout(0.15))
model.add(Bidirectional(LSTM(200, return_sequences=False, kernel_regularizer=l2(0.0001))))
model.add(Dropout(0.15))
model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.0001)))
model.add(Dropout(0.15))
model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.0001)))
model.add(Dense(1))

opt = Adam(learning_rate=0.0001, clipnorm=1.0)
model.compile(optimizer=opt, loss='mean_squared_error')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train_scaled, 
    epochs=50, 
    batch_size=32, 
    validation_data=(X_test_scaled, y_test_scaled), 
    callbacks=[reduce_lr, early_stop]
)

test_loss = model.evaluate(X_test_scaled, y_test_scaled)
print(f'Final Test Loss: {test_loss}')

pred_y_scaled = model.predict(X_test_scaled)
pred_y_scaled = pred_y_scaled.reshape(-1, 1)

pred_y = scaler_y.inverse_transform(pred_y_scaled)
y_test_inv = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

if pred_y.shape == y_test_inv.shape:
    rmse = np.sqrt(np.mean((pred_y - y_test_inv)**2))
    print(f'Final Test RMSE: {rmse}')
    correlation_coefficient = np.corrcoef(pred_y.flatten(), y_test_inv.flatten())[0, 1]
    print(f'Correlation Coefficient: {correlation_coefficient}')
else:
    print("Shape mismatch between predictions and test labels.")

