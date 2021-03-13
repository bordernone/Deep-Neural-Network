import torch
import torchvision
from torch import nn
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset_train = torchvision.datasets.MNIST('MNIST/', train=True, download=True, transform=transform)

print(dataset_train.data.shape)
print(dataset_train.targets.shape)

dataset_test = torchvision.datasets.MNIST('MNIST/', train=False, download=False, transform=transform)

print(dataset_test.data.shape)
print(dataset_test.targets.shape)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=200, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=200, shuffle=False)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 10, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.dense = nn.Sequential(
            nn.Linear(6 * 6 * 20, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 6 * 6 * 20)
        return self.dense(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == labels).float())


net = Model()
net.to(device)

optimizer = torch.optim.Adadelta(net.parameters(), lr=0.01)
num_epochs = 30
loss_fn = nn.CrossEntropyLoss()

for i in range(num_epochs):
    net.train()
    loss_per_epoch = 0.00
    accuracy_per_epoch = 0.00
    for data, labels in iter(train_loader):
        data, labels = data.to(device), labels.to(device)
        predictions = net(data)
        optimizer.zero_grad()
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
        loss_per_epoch += loss.item()
        accuracy_per_epoch += accuracy(predictions, labels)
    loss_per_epoch /= len(train_loader)
    accuracy_per_epoch /= len(train_loader)

    net.eval()
    with torch.no_grad():
        test_accuracy = 0.00
        for data, labels in iter(test_loader):
            data, labels = data.to(device), labels.to(device)
            predictions = net(data)
            test_accuracy += accuracy(predictions, labels)
        test_accuracy /= len(test_loader)

    print(f"Epoch {i}, loss = {loss_per_epoch}, accuracy = {accuracy_per_epoch}, test_accuracy = {test_accuracy}")
