# predict.py
import torch
from lstm_model import LSTMModel
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the trained LSTM model
lstm_model = LSTMModel(input_size=1, hidden_layer_size=100, output_size=1)
lstm_model.load_state_dict(torch.load('saved_lstm_model.pt'))
lstm_model.eval()

# Load new data for time series predictions
new_data = pd.read_csv('data/new_time_series_data.csv')  # Update with your new dataset file

# Convert the 'timestamp' column to datetime format
new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])

# Set the 'timestamp' column as the index
new_data.set_index('timestamp', inplace=True)

# Normalize the 'value' column using Min-Max scaling (using the same scaler from training)
scaler = MinMaxScaler()
new_data['value_scaled'] = scaler.fit_transform(new_data[['value']])

# Convert the scaled values to PyTorch tensor
input_data = torch.tensor(new_data['value_scaled'].values, dtype=torch.float32).view(-1, 1, 1)

# Make time series predictions
with torch.no_grad():
    lstm_model.eval()
    predicted_values = []

    # Initialize hidden state and cell state
    hidden_cell = (torch.zeros(1, 1, lstm_model.hidden_layer_size),
                   torch.zeros(1, 1, lstm_model.hidden_layer_size))

    for i in range(len(input_data)):
        output, hidden_cell = lstm_model(input_data[i].view(1, 1, -1), hidden_cell)
        predicted_values.append(output.item())

# Denormalize the predicted values
predicted_values = scaler.inverse_transform([predicted_values])

# Create a DataFrame with the predicted values and corresponding timestamps
predicted_df = pd.DataFrame({'timestamp': new_data.index, 'predicted_value': predicted_values[0]})

# Display the predicted values
print("\nPredicted Values:")
print(predicted_df.head())
