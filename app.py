import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Dense
from keras.callbacks import EarlyStopping
from math import sqrt
import os

# -----------------------------
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -----------------------------
# Streamlit page setup
st.set_page_config(page_title="Electricity Demand Forecasting", layout="wide")
st.title("⚡ Electricity Demand Forecasting using RNN")
st.markdown("Forecast future electricity **power consumption (MW)** using historical data and an RNN model.")

# -----------------------------
# File upload
uploaded_file = st.file_uploader("Upload CSV file (with 'Datetime' & 'AEP_MW' columns)", type=["csv"])

# -----------------------------
# Dataset creation (vectorized)
def create_dataset(dataset, time_step=24):
    n_samples = len(dataset) - time_step
    if n_samples <= 0:
        return np.array([]), np.array([])
    X = np.lib.stride_tricks.sliding_window_view(dataset, window_shape=(time_step, 1))[:-1, :, 0]
    y = dataset[time_step:, 0]
    return X, y

# -----------------------------
# Preprocess data (cached)
@st.cache_data(show_spinner=False)
def preprocess_data(df, time_step=24, limit_rows=2000):
    # Automatically parse datetime (works with ISO format)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').set_index('Datetime')
    df = df.tail(limit_rows)  # Limit rows for faster run
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[['AEP_MW']])
    X, y = create_dataset(scaled, time_step)
    return df, scaled, scaler, X, y

# -----------------------------
# Load RNN model (cached)
@st.cache_resource(show_spinner=False)
def load_rnn_model(path="model_rnn.keras"):
    if os.path.exists(path):
        return load_model(path)
    return None

# -----------------------------
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if not {'Datetime', 'AEP_MW'}.issubset(data.columns):
        st.error("CSV must contain 'Datetime' and 'AEP_MW' columns!")
    else:
        st.subheader("📊 Raw Data Preview")
        st.write(data.head())

        # -----------------------------
        # Dataset parameters
        time_step = st.slider("Select time steps (hours)", 12, 72, 24)
        data, scaled_data, scaler, X, y = preprocess_data(data, time_step=time_step)

        if X.size == 0:
            st.warning("⚠️ Not enough data for the selected time step. Upload more data or reduce time step.")
        else:
            # Plot raw data
            st.subheader("📈 Historical Power Consumption")
            st.line_chart(data['AEP_MW'])

            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Reshape for RNN input
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # -----------------------------
            # Train model
            st.subheader("🚀 Train RNN Model")
            epochs = st.slider("Select number of epochs", 10, 100, 50)
            batch_size = st.slider("Select batch size", 16, 128, 32)

            if st.button("Train Model"):
                model = Sequential([
                    SimpleRNN(32, activation='tanh', input_shape=(time_step, 1)),
                    Dense(16, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                with st.spinner("Training model... please wait ⏳"):
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=batch_size,
                        verbose=0, callbacks=[callback]
                    )
                    model.save("model_rnn.keras")

                st.success("✅ Model training completed and saved as model_rnn.keras")

                # Predict and metrics
                y_pred = model.predict(X_test)
                y_predicted = scaler.inverse_transform(y_pred)
                y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

                mae = mean_absolute_error(y_actual, y_predicted)
                rmse = sqrt(mean_squared_error(y_actual, y_predicted))

                st.subheader("📏 Model Performance Metrics")
                st.write(f"**Mean Absolute Error (MAE):** {mae:.2f} MW")
                st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f} MW")

                # Plot results
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(y_actual[:200], label='Actual Power Consumption (MW)')
                ax.plot(y_predicted[:200], label='Predicted Power Consumption (MW)')
                ax.set_title("Actual vs Predicted Power Consumption")
                ax.set_xlabel("Time Steps")
                ax.set_ylabel("Power (MW)")
                ax.legend()
                st.pyplot(fig)

                # Loss curve
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                ax2.plot(history.history['loss'], label='Training Loss')
                ax2.plot(history.history['val_loss'], label='Validation Loss')
                ax2.set_title("Training vs Validation Loss")
                ax2.set_xlabel("Epochs")
                ax2.set_ylabel("Loss")
                ax2.legend()
                st.pyplot(fig2)

            # -----------------------------
            # Forecast future hours
            st.subheader("🔮 Forecast Future Power Consumption")
            future_hours = st.slider("Select number of hours to forecast", 1, 24, 1)

            if st.button("Forecast Next Hours"):
                model = load_rnn_model()
                if model is None:
                    st.error("Model not found! Please train the model first.")
                else:
                    input_seq = scaled_data[-time_step:].reshape(1, time_step, 1)
                    forecasts = []

                    for _ in range(future_hours):
                        pred = model.predict(input_seq, verbose=0)
                        forecasts.append(pred[0][0])
                        input_seq = np.concatenate([input_seq[:, 1:, :], pred.reshape(1, 1, 1)], axis=1)

                    forecasts_rescaled = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))

                    forecast_df = pd.DataFrame({
                        'Datetime': pd.date_range(
                            start=data.index[-1] + pd.Timedelta(hours=1),
                            periods=future_hours, freq='H'
                        ),
                        'Predicted Power Consumption (MW)': forecasts_rescaled.flatten()
                    })

                    st.table(forecast_df)
                    st.download_button(
                        "📥 Download Forecast",
                        data=forecast_df.to_csv(index=False),
                        file_name="forecast_power_consumption.csv"
                    )

else:
    st.info("👆 Please upload a CSV file to begin forecasting.")
