import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# dataset


# Fetching Ethereum prices
def fetch_ethereum_prices():
    # Filename to check if the data is already downloaded
    filename = 'data/ethereum_prices_last_30_days.csv'

    # Check if the file already exists
    if not os.path.isfile(filename):
        # URL for the API to fetch Ethereum prices
        api_url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
        
        # Dates for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        # Parameters for the API request
        params = {
            'vs_currency': 'usd',
            'days': '720',
            'interval': 'daily'
        }

        # Making the API request
        response = requests.get(api_url, params=params)

        if response.status_code == 200:
            # Parsing the JSON response
            data = response.json()
            prices = data['prices']

            # Creating a DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Saving the DataFrame to a CSV file
            df.to_csv(filename, index=False)

            return df
        else:
            return f"Failed to fetch data: {response.status_code}"
    else:
        # Load the data from the CSV file
        return pd.read_csv(filename)

ethereum_prices = fetch_ethereum_prices()
# print(ethereum_prices.head())

# Assuming ethereum_prices is a normalized Pandas DataFrame of your data
data = torch.tensor(ethereum_prices['price'], dtype=torch.float32)
print(data.shape)

# reshape data to (batch_size, input_size)
data = data.view(-1, 1)
print(data.shape)

# exit(0)
#plt.plot(ethereum_prices['price'])
# plt.show()



# Hyper-parameters 
# input_size = 784 # 28x28
# hidden_size = 500 
# num_classes = 10
# num_epochs = 2
# batch_size = 100
# learning_rate = 0.001

# define the model
class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__() 
        # self.layers = [
        #     nn.LSTM(hidden_size=64, input_size=1, num_layers=1, 
        #         bidirectional=False, device=device, dropout=0.0
        #     )
        # ]

        self.lstm = nn.LSTM(hidden_size=64, input_size=1, num_layers=1, 
                bidirectional=False, device=device, dropout=0.0
            )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out = self.lstm(x)
        out = self.fc(out)

        return out
        # for layer in self.layers:
        #     x = layer(x)
        
        # return x


model = MyLSTM().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# train the model
num_epochs = 2
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=False)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (input) in enumerate(train_loader):  
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 28*28).to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         # max returns (value ,index)
#         _, predicted = torch.max(outputs.data, 1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()

#     acc = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network on the 10000 test images: {acc} %') 




