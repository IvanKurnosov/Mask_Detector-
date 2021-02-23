import torch


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=5, padding=0)
        self.act1 = torch.nn.ELU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1ad = torch.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, padding=2)
        self.act1ad = torch.nn.ELU()
        self.pool1ad = torch.nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv2 = torch.nn.Conv2d(
            in_channels=24, out_channels=48, kernel_size=5, padding=0)
        self.act2 = torch.nn.ELU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2ad = torch.nn.Conv2d(in_channels=48, out_channels=96, kernel_size=5, padding=2)
        self.act2ad = torch.nn.ELU()
        self.pool2ad = torch.nn.MaxPool2d(kernel_size=2, stride=1)

        self.fc1 = torch.nn.Linear(3 * 3 * 96, 120)
        self.act3 = torch.nn.ELU()

        self.fc2 = torch.nn.Linear(120, 84)
        self.act4 = torch.nn.ELU()

        self.fc3 = torch.nn.Linear(84, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv1ad(x)
        x = self.act1ad(x)
        x = self.pool1ad(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv2ad(x)
        x = self.act2ad(x)
        x = self.pool2ad(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)

        return x


