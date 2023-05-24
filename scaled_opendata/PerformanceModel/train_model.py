import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from model import PerforamnceCNN


# Define the CNN model
model = PerforamnceCNN()

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the training data
train_data = data.TensorDataset(torch.randn(1000, 20, 50), torch.randint(0, 2, (1000,)))
train_loader = data.DataLoader(train_data, batch_size=100)

# Train the model
for epoch in range(10):
    for i, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)

        # Calculate the loss
        loss = criterion(output, target)

        # Backpropagate the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss
        if i % 100 == 0:
            print(loss.item())

# Save the model
torch.save(model.state_dict(), "model.pt")