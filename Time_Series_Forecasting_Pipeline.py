import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import streamlit as st

# Initialize Streamlit page configuration
st.set_page_config(page_title="Advanced Time Series Forecasting with LSTM", layout="wide")

# Function to clean and prepare data
def clean_and_prepare_data(df, window_size=12, value_column_name=None):
    # If the column name isn't provided, assume the first column is the value column
    if value_column_name is None:
        value_column_name = df.columns[0]
    
    # Drop any non-numeric columns (e.g., date columns)
    df = df.select_dtypes(include=[np.number])
    
    # Extract the target column values
    values = df[value_column_name].values.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)
    
    X, y = [], []
    
    # Create sequences of data for LSTM
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])  # Window of previous data points
        y.append(scaled_data[i, 0])  # The next data point to predict
    
    X, y = np.array(X), np.array(y)
    
    # Reshaping X for LSTM input
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler, value_column_name

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_layer_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

# Training function
def train_lstm_model(X, y, epochs=100, batch_size=32, lr=0.001):
    # Convert to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch.unsqueeze(1))  # Reshape y_batch to match output
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 20 == 0:
            st.write(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return model

# Fixed forecasting function
def forecast_lstm(model, data_scaled, window_size=12, future_steps=12):
    model.eval()
    last_sequence = torch.tensor(data_scaled[-window_size:], dtype=torch.float32).unsqueeze(0)
    
    predictions = []
    with torch.no_grad():
        for _ in range(future_steps):
            pred = model(last_sequence)
            predictions.append(pred.item())
            
            # Correct 3D tensor reshaping
            new_element = pred.unsqueeze(1)  # [batch_size=1, seq_len=1, features=1]
            
            # Maintain 3D shape during update
            last_sequence = torch.cat(
                (last_sequence[:, 1:, :], new_element),
                dim=1
            )

    return np.array(predictions)

# Visualization function for future predictions with prediction interval
def plot_future_predictions_with_interval(df, future_predictions_rescaled, value_column_name):
    # Define a simple prediction interval (e.g., ±10% margin)
    lower_bound = future_predictions_rescaled - (0.1 * future_predictions_rescaled)
    upper_bound = future_predictions_rescaled + (0.1 * future_predictions_rescaled)
    
    trace_actual = go.Scatter(
        x=df.index,
        y=df[value_column_name],
        mode='lines',
        name='Actual Data'
    )

    trace_pred = go.Scatter(
        x=np.arange(len(df), len(df) + len(future_predictions_rescaled)),
        y=future_predictions_rescaled.flatten(),
        mode='lines',
        name='Predicted Data',
        line=dict(dash='dot', color='orange')
    )

    trace_upper = go.Scatter(
        x=np.arange(len(df), len(df) + len(future_predictions_rescaled)),
        y=upper_bound.flatten(),
        mode='lines',
        name='Upper Bound',
        line=dict(dash='dashdot', color='gray')
    )

    trace_lower = go.Scatter(
        x=np.arange(len(df), len(df) + len(future_predictions_rescaled)),
        y=lower_bound.flatten(),
        mode='lines',
        name='Lower Bound',
        line=dict(dash='dashdot', color='gray')
    )
    
    layout = go.Layout(
        title='Predicted Data with Confidence Interval',
        xaxis=dict(title='Date'),
        yaxis=dict(title=value_column_name),
        showlegend=True
    )
    
    fig = go.Figure(data=[trace_actual, trace_pred, trace_upper, trace_lower], layout=layout)
    st.plotly_chart(fig)

# Rolling statistics visualization
def plot_rolling_statistics(df, window_size, value_column_name):
    rolling_mean = df[value_column_name].rolling(window=window_size).mean()
    rolling_std = df[value_column_name].rolling(window=window_size).std()

    trace_actual = go.Scatter(
        x=df.index,
        y=df[value_column_name],
        mode='lines',
        name='Actual Data'
    )

    trace_rolling_mean = go.Scatter(
        x=df.index,
        y=rolling_mean,
        mode='lines',
        name=f'Rolling Mean ({window_size} days)',
        line=dict(color='green')
    )

    trace_rolling_std = go.Scatter(
        x=df.index,
        y=rolling_std,
        mode='lines',
        name=f'Rolling Std Dev ({window_size} days)',
        line=dict(color='red')
    )

    layout = go.Layout(
        title=f'Rolling Statistics with Window Size {window_size} Days',
        xaxis=dict(title='Date'),
        yaxis=dict(title=value_column_name),
        showlegend=True
    )

    fig = go.Figure(data=[trace_actual, trace_rolling_mean, trace_rolling_std], layout=layout)
    st.plotly_chart(fig)

# Streamlit interface
st.title("Advanced Time Series Forecasting with LSTM")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the dataset
    st.subheader("Uploaded Data")
    st.write(df.head())
    
    try:
        # Ensure we handle date columns (drop them for scaling)
        if 'Date' in df.columns:  # Assuming 'Date' is the column name for dates
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)  # Set 'Date' as the index

        X, y, scaler, value_column_name = clean_and_prepare_data(df)
        model = train_lstm_model(X, y)
        
        # Get scaled data for forecasting
        data_scaled = scaler.transform(df[[value_column_name]].values.reshape(-1, 1))
        
        # Forecast
        future_steps = 12  # Adjust number of future steps to predict
        predictions = forecast_lstm(model, data_scaled, future_steps=future_steps)
        predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))
        
        # Display results
        plot_future_predictions_with_interval(df, predictions_rescaled, value_column_name)
        
        # Display the future predicted values
        future_dates = pd.date_range(df.index[-1], periods=len(predictions_rescaled) + 1, freq='D')[1:]
        future_df = pd.DataFrame(data=predictions_rescaled, index=future_dates, columns=[value_column_name])
        st.subheader("Future Predicted Values")
        st.write(future_df)
        
        # Plot rolling statistics
        rolling_window_size = 30  # 30-day rolling window for statistics
        plot_rolling_statistics(df, rolling_window_size, value_column_name)
        
    except ValueError as e:
        st.error(str(e))
