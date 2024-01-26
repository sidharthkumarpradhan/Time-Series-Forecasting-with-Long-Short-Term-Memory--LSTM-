# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from lstm_model import LSTMModel
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Add code to set up data transformations, create DataLoader, and define LSTM model

# Initialize LSTM model, optimizer, and criterion
lstm_model = LSTMModel(input_size=1, hidden_layer_size=100, output_size=1)
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Move model to device
lstm_model = lstm_model.to(device)
criterion = criterion.to(device)

# Load the preprocessed time series data
preprocessed_data = pd.read_csv('data/preprocessed_time_series_data.csv')  # Update with your preprocessed dataset file

# Normalize the 'value' column using Min-Max scaling
scaler = MinMaxScaler()
preprocessed_data['value_scaled'] = scaler.fit_transform(preprocessed_data[['value']])

# Convert the scaled values to PyTorch tensor
input_data = torch.tensor(preprocessed_data['value_scaled'].values, dtype=torch.float32).view(-1, 1, 1)

# Convert the target values to PyTorch tensor
target_data = torch.tensor(preprocessed_data['value_scaled'].values, dtype=torch.float32).view(-1, 1, 1)

# Combine input and target sequences into DataLoader
dataset = torch.utils.data.TensorDataset(input_data, target_data)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Training loop
def train_lstm(model, optimizer, criterion, data_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for i, (input_seq, target_seq) in enumerate(data_loader):
            optimizer.zero_grad()

            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            # Forward pass
            output_seq = model(input_seq)

            # Compute the loss
            loss = criterion(output_seq, target_seq)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print the average loss for the epoch
        average_loss = total_loss / len(data_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

# Set the number of training epochs
num_epochs = 50

# Train the LSTM model
train_lstm(lstm_model, optimizer, criterion, data_loader, num_epochs)

# Save the trained model
torch.save(lstm_model.state_dict(), 'saved_lstm_model.pt')
