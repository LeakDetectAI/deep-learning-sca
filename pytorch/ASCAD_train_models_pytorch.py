import os
import os.path
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

sys.path.append('/content/drive/MyDrive/smooth-topk')

from topk.svm import SmoothTopkSVM


class CNNBest(nn.Module):
    def __init__(self, classes=256, input_dim=700):
        super().__init__()

        self.classes = classes
        self.input_dim = input_dim

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, padding='same')
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, padding='same')
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=11, padding='same')
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=11, padding='same')
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=11, padding='same')

        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=21 * 512, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc_final = nn.Linear(in_features=4096, out_features=self.classes)

    def forward(self, x):
        # Block 1
        x = self.avg_pool(F.relu(self.conv1(x)))
        # Block 2
        x = self.avg_pool(F.relu(self.conv2(x)))
        # Block 3
        x = self.avg_pool(F.relu(self.conv3(x)))
        # Block 4
        x = self.avg_pool(F.relu(self.conv4(x)))
        # Block 5
        x = self.avg_pool(F.relu(self.conv5(x)))
        # Classification Block
        x = x.view(-1, 21 * 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc_final(x))

        return x

    def get_classes(self):
        return self.classes


class ASCADDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.n_samples = X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.n_samples


def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()


def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


def load_ascad(ascad_database_file, load_metadata=False):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (
            in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])


def topk_categorical_accuracy_np(k=5, normalize=False):
    def topk_acc(y_true, y_pred):
        n_objects = y_pred.shape[-1]
        topK = y_pred.argsort(axis=1)[:, -k:][:, ::-1]
        accuracies = np.zeros_like(y_true, dtype=bool)
        # y_true = np.argmax(y_true, axis=1)
        for i, top in enumerate(topK):
            accuracies[i] = y_true[i] in top
        accuracies = np.mean(accuracies)
        if normalize:
            minimum = k / n_objects
            accuracies = (accuracies - minimum) / (1.0 - minimum)
        return accuracies

    return topk_acc


def evaluate_model(scores, y_test, top1, top3, top5, top10):
    t1, t3, t5, t10 = top1(y_test, scores), top3(y_test, scores), top5(y_test, scores), top10(y_test, scores)
    return {"Top1": t1, "Top3": t3, "Top5": t5, "Top10": t10}


def calculate_mean_rank(y_true, final_scores):
    scores_df = pd.DataFrame(data=final_scores)
    final_ranks = scores_df.rank(ascending=False, axis=1)
    final_ranks = final_ranks.to_numpy(dtype='int32')
    predicted_ranks = np.zeros(shape=(y_true.shape[0]))
    for itr in range(y_true.shape[0]):
        true_label = y_true[itr]
        predicted_ranks[itr] = final_ranks[itr, true_label]
    return np.mean(predicted_ranks)


def train(no_of_epochs, model, dataloader_train, optimizer, loss_fn, train_set_size, device):
    model.to(device)
    summary(model, (1, 700))
    # loop over the dataset multiple times
    for epoch in range(no_of_epochs):
        running_loss = 0.0
        correct = 0
        for i, data in enumerate(dataloader_train, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # print(inputs)
            # print(type(inputs))
            inputs, labels = inputs.to(device), labels.to(device)
            # print(inputs)
            # print(type(inputs))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # add correctly predicted examples and loss for the current batch
            # labels_one_hot = F.one_hot(labels, num_classes=256)
            predicted_classes = torch.argmax(outputs, dim=1)
            # true_classes = torch.argmax(labels, dim=1)
            true_classes = labels
            correct += (predicted_classes == true_classes).float().sum()
            running_loss += loss.item()

        accuracy = correct.item() / train_set_size
        print('Epoch : ' + str(epoch + 1) + ', Loss : ' + str(running_loss) + ', Accuracy : ' + str(round(accuracy, 4)))

    print('Training Completed')


def test(model, dataloader_test, loss_fn, Y_attack_np, test_set_size, num_classes, batch_size, device):
    correct = 0
    total = 0
    running_loss = 0.0

    index = 0
    final_prob_predictions = np.zeros((test_set_size, num_classes))

    # no gradients for testing the model
    with torch.no_grad():
        for data in dataloader_test:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # outputs received by running the inputs through the network
            outputs = model(inputs)

            final_prob_predictions[index:index + batch_size] = outputs.cpu().detach().numpy()
            index += batch_size

            # max returns a tuple (values, indices), indices are the predicted classes (argmax)
            _, predicted = torch.max(outputs.data, 1)

            # calculate loss on the current batch
            loss = loss_fn(outputs, labels)

            total += labels.size(0)
            correct += (predicted == labels).float().sum()
            running_loss += loss.item()

    # Score
    accuracy = correct.item() / total
    print('Loss : ' + str(running_loss) + ', Accuracy : ' + str(round(accuracy, 4)))

    # Topk
    top1 = topk_categorical_accuracy_np(k=1, normalize=False)
    top5 = topk_categorical_accuracy_np(k=5, normalize=False)
    top3 = topk_categorical_accuracy_np(k=3, normalize=False)
    top10 = topk_categorical_accuracy_np(k=10, normalize=False)

    topk_results = evaluate_model(final_prob_predictions, Y_attack_np, top1, top3, top5, top10)
    print('Topk : ' + str(topk_results))

    # Mean Rank
    attack_mean_rank = calculate_mean_rank(Y_attack_np, final_prob_predictions)
    print('Mean Rank : ' + str(attack_mean_rank))


no_of_epochs = 75
num_classes = 256
batch_size = 200

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = CNNBest()
# categorical_cross_entropy, nn.CrossEntropyLoss()
loss = SmoothTopkSVM(n_classes=num_classes, k=10)
loss.cuda(device=device)
optimizer = optim.RMSprop(model.parameters(), lr=0.00001)

ascad_database = '/content/drive/MyDrive/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD_desync0.h5'
PATH = '/content/drive/MyDrive/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_pytorch.pth'

# ascad_database = '/content/drive/MyDrive/ASCAD/ASCAD_desync0.h5'
# PATH = '/content/drive/MyDrive/ASCAD/cnn_best_pytorch.pth'

(X_profiling, Y_profiling), (X_attack, Y_attack) = load_ascad(ascad_database)

X_attack_np = X_attack
Y_attack_np = Y_attack

# training & saving the model
file_path = os.path.normpath(PATH)
if not os.path.exists(file_path):
    X_profiling = torch.from_numpy(
        X_profiling.reshape((X_profiling.shape[0], 1, X_profiling.shape[1])).astype('float32'))
    # Y_profiling = torch.eye(num_classes, dtype=torch.int32)[Y_profiling]
    Y_profiling = torch.from_numpy(Y_profiling)
    dataset_train = ASCADDataset(X_profiling, Y_profiling)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

    train(no_of_epochs=no_of_epochs, model=model, dataloader_train=dataloader_train,
          optimizer=optimizer, loss_fn=loss, train_set_size=X_profiling.shape[0], device=device)
    torch.save(model.state_dict(), PATH)
else:
    model.load_state_dict(torch.load(PATH))
    model.eval()
    model.to(device)

# testing the model
X_attack = torch.from_numpy(X_attack.reshape((X_attack.shape[0], 1, X_attack.shape[1])).astype('float32'))
# Y_attack = torch.eye(num_classes, dtype=torch.int32)[Y_attack]
Y_attack = torch.from_numpy(Y_attack)
dataset_test = ASCADDataset(X_attack, Y_attack)
dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=2)

test(model=model, dataloader_test=dataloader_test, loss_fn=loss, Y_attack_np=Y_attack_np,
     test_set_size=X_attack_np.shape[0], num_classes=num_classes, batch_size=batch_size, device=device)
