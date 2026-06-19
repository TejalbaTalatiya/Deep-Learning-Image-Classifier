import torch
import torch.nn as nn
import torch.optim as optim

class CoreConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(CoreConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 16 * 16, 2) 

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = CoreConvolutionalNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    mock_images = torch.randn(5, 3, 32, 32)
    mock_labels = torch.randint(0, 2, (5,))
    
    outputs = model(mock_images)
    loss = criterion(outputs, mock_labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"[Success] Initial Deep Learning Training Loop Executed. Training Loss: {loss.item():.4f}")
