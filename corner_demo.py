import json
from types import SimpleNamespace
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from vgg import *
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch


class cornerDataset(Dataset):

    def __init__(self, datasetPath, mode):
        self.dataPath = datasetPath
        self.datasetNames = os.listdir(self.dataPath)
        self.mode = mode

    def __len__(self):
        return len(self.datasetNames)

    def __getitem__(self, idx):
        name = self.datasetNames[idx]

        img = np.load(self.dataPath + "/" + name)

        ### normalizing
        img[:, :, 0] /= np.max(img[:, :, 0])
        img[:, :, 1] /= np.max(img[:, :, 1])



        if self.mode == "train":
            if np.random.random() > 0.5:
                ## flipping
                img[:, :, 0] = np.flip(img[:, :, 0])
                img[:, :, 0] = np.flip(img[:, :, 0])

        if name[-14:-11] == "non":
            label = 0
        else:
            label = 1

        img = np.moveaxis(img, [2, 1], [0, 2]).astype(np.float32)

        return label, img


def find_corner_index(utc_time_corner, current_track):

    for index in range(len(current_track)-1):

        current = datetime.datetime.utcfromtimestamp(current_track[index]["utc_time"] / 1000.0)
        next = datetime.datetime.utcfromtimestamp(current_track[index+1]["utc_time"] / 1000.0)
        diff = millis_interval(current, next)

        if current<= utc_time_corner and utc_time_corner <= next:

            return index



### stackoverflow
def millis_interval(start, end):
    """start and end are datetime instances"""
    diff = end - start
    millis = diff.days * 24 * 60 * 60 * 1000
    millis += diff.seconds * 1000
    millis += diff.microseconds / 1000
    return millis

def draw_players(instance, img, M, N):

    for player in instance:
        x = min(int(N / 2 + player["position"][0]), N - 1)
        x = max(x, 0)

        y = min(int(M / 2 + player["position"][1]), M - 1)
        y = max(y, 0)

        img[y, x] += 1

    return img

def constract_dataset(tracking_data_folder, event_dataset_folder, M, N, non_corner_mul):


    jsonInstances = os.listdir(tracking_data_folder)


    if not os.path.exists(event_dataset_folder + "Train"):
        os.makedirs(event_dataset_folder + "Train")

    if not os.path.exists(event_dataset_folder + "Val"):
        os.makedirs(event_dataset_folder + "Val")


    for jsonIndex, jsonFile in enumerate(jsonInstances[3::4]):

        print(jsonIndex, "/", len(jsonInstances[3::4]))

        game_id = jsonFile.split(".")[0]
        phase = jsonFile.split(".")[1][0]
        ## tracks ### use only tracking data
        f = open(tracking_data_folder + jsonFile)
        current_track = json.load(f)

        f = open('corner-detection-challenge.json')
        ground_truth = json.load(f)["game_id"][game_id][phase]

        ### Constructing corner sample
        corner_indices = []
        created_corners = 0

        for corner_index in range(len(ground_truth)):

            try:
                utc_time_corner = datetime.datetime.strptime(ground_truth[corner_index]["utc_time"][:-6], '%Y-%m-%d %H:%M:%S.%f')
            except:
                utc_time_corner = datetime.datetime.strptime(ground_truth[corner_index]["utc_time"][:-6][:-4], '%Y-%m-%d %H:%M:%S.%f')

            corner_idx = find_corner_index(utc_time_corner, current_track)

            corner_indices.append(corner_idx)

            for d in [i for i in range(-20, 40, 20)]:

                corner_index_start = max(corner_idx + d - 50, 0) ## 2 seconds before the corner
                corner_index_end = min(corner_idx + d + 50, len(current_track)-1) ## 2 seconds after the corner

                img = np.zeros([M, N, 2])


                for index, instance in enumerate(current_track[corner_index_start:corner_index_end]):

                    img[:, :, 0] = draw_players(instance['away_team'], img[:, :, 0], M, N)
                    img[:, :, 1] = draw_players(instance['home_team'],  img[:, :, 1], M, N)

                    if phase == "1":
                        save_name = event_dataset_folder+"Train/"
                    else:
                        save_name = event_dataset_folder+"Val/"

                    save_name += game_id+phase+str(corner_index)+"-"+str(d)+"-corner.npy"

                if np.max(img) > 0: ## only if actual instance existed
                    np.save(save_name, img)
                    created_corners += 1

        ## not too early not late
        non_corner_indices = random.sample(range(500, len(current_track)-500), created_corners*(non_corner_mul+1)) ## number of non-corner events equal to the number of corners
        created_non_corners = 0

        for curret_non_corner_index in non_corner_indices:

            corner_index_start = curret_non_corner_index - 50 ## 2 seconds before the corner
            corner_index_end = curret_non_corner_index + 50 ## 2 seconds after the corner

            img = np.zeros([M, N, 2])

            for index, instance in enumerate(current_track[corner_index_start:corner_index_end]):


                img[:, :, 0] = draw_players(instance['away_team'], img[:, :, 0], M, N)
                img[:, :, 1] = draw_players(instance['home_team'], img[:, :, 1], M, N)

                if phase == "1":
                    save_name = event_dataset_folder + "Train/"
                else:
                    save_name = event_dataset_folder + "Val/"

            if not(np.sum([abs(i-curret_non_corner_index) < 500 for i in corner_indices])) and created_non_corners < created_corners*non_corner_mul:
                save_name += game_id + phase + str(curret_non_corner_index) + "-non_corner.npy"
                np.save(save_name, img)  # sufficiently far away from corner instance
                created_non_corners += 1

            else:
                pass ## not safe to consider it as corner
    return


### Setting arguments
args = SimpleNamespace(tracking_data_folder='Data/Tracking Data/',
                       event_dataset_folder="Event_dataset/",
                       M=68,
                       N=105,
                       batch_size=2,
                       non_corner_mul=1,
                       epochs=8,
                       lr=0.001,
                       weight_decay=1e-04)


## construct dataset
constract_dataset(args.tracking_data_folder, args.event_dataset_folder, args.M, args.N, args.non_corner_mul)

model = vgg19_bn()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

train_set = cornerDataset(datasetPath=args.event_dataset_folder+"Train", mode="train")
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)


val_set = cornerDataset(datasetPath=args.event_dataset_folder+"Val", mode="val")
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)


train_loss_history = np.zeros(args.epochs)
val_loss_history = np.zeros(args.epochs)
train_acc_history = np.zeros(args.epochs)
val_acc_history = np.zeros(args.epochs)

for epoch in range(args.epochs):

    ### Training
    model.train()
    confussion_matrix_train = np.zeros([2,2])


    for train_index, (label, data) in enumerate(train_loader):
        # print("~Training:", train_index, "/", len(train_loader))

        x = model(data)
        loss = criterion(x, label)

        # updating weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.max(x, 1)[1]

        for p in range(predictions.shape[0]):
            confussion_matrix_train[predictions[p].data.cpu().numpy(), label[p].data.cpu().numpy()] += 1

        acurracy = torch.sum(predictions == label)/args.batch_size

        train_loss_history[epoch] += loss
        train_acc_history[epoch] += acurracy

    train_loss_history[epoch] /= len(train_loader)
    train_acc_history[epoch] /= len(train_loader)

    ### Validating
    model.eval()
    confussion_matrix_val = np.zeros([2,2])

    for val_index, (label, data) in enumerate(val_loader):
        # print("~Validating:", val_index, "/", len(val_loader))
        x = model(data)
        loss = criterion(x, label)

        predictions = torch.max(x, 1)[1]

        for p in range(predictions.shape[0]):
            confussion_matrix_val[predictions[p].data.cpu().numpy(),label[p].data.cpu().numpy()] += 1


        acurracy = torch.sum(predictions == label) / args.batch_size

        val_loss_history[epoch] += loss
        val_acc_history[epoch] += acurracy

    val_loss_history[epoch] /= len(val_loader)
    val_acc_history[epoch] /= len(val_loader)

    ### Printing epoch results
    print('Epoch: {}/{} \n'
          'Train ~ loss: {:.2f} acc: {:.2f}\n'
          'Val ~ loss: {:.2f} acc: {:.2f}\n'.format(epoch, args.epochs-1,
                                                    train_loss_history[epoch], train_acc_history[epoch],
                                                    val_loss_history[epoch], val_acc_history[epoch]))


plt.figure()
plt.title("Losses")
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Val Loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.savefig('Losses.png', bbox_inches='tight')

plt.figure()
plt.title("Accuracies")
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(val_acc_history, label="Val Accuracies")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.savefig('Accuracies.png', bbox_inches='tight')
