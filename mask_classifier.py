import torch
import numpy as np
import random
from face_dataset import Dataset
from face_dataset import image_prepare
from network import Network


WEIGHT_PATH = 'weights.pt'


class MaskClassifier():
    def __init__(self):
        self.network = Network()
        self.network.load_state_dict(torch.load(WEIGHT_PATH))

    def is_in_mask(self, face):
        face = torch.tensor([image_prepare(face)]).float()
        result = self.network.forward(face).argmax(dim=1)[0].bool()
        return result


def train():
    dataset = Dataset()
    X_train = dataset.train_data.float()
    y_train = dataset.train_labels.long()
    X_test = dataset.test_data.float()
    y_test = dataset.test_labels.long()

    network = Network()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network = network.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1.0e-4)

    batch_size = 10

    test_accuracy_history = []
    test_loss_history = []

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    max_accuracy = 0
    for epoch in range(10):
        order = np.random.permutation(len(X_train))
        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()

            batch_indexes = order[start_index:start_index + batch_size]

            X_batch = X_train[batch_indexes].to(device)
            y_batch = y_train[batch_indexes].to(device)

            preds = network.forward(X_batch)

            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()

        test_preds = network.forward(X_test)
        test_loss_history.append(loss(test_preds, y_test).data.cpu().tolist())

        accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()
        test_accuracy_history.append(accuracy.tolist())

        if accuracy > max_accuracy:
            torch.save(network.state_dict(), WEIGHT_PATH)

        #print(accuracy.data)


    print(test_accuracy_history)
    print(test_loss_history)

