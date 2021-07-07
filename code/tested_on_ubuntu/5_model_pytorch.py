import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import torch.nn.functional as F
import os
import signal
import sys
import torch.nn as nn
from tqdm import trange
import time
import matplotlib.pyplot as plt
from torch.autograd import Variable
from matplotlib.lines import Line2D
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sn
import random

# torch.autograd.set_detect_anomaly(True)

giwrgos = False
train_set_path = ''
val_set_path = ''
test_set_path = ''
train_landm_csvs_path = ''
val_landm_csvs_path = ''
test_landm_csvs_path = ''
root_video_path = r'../GSL_isol'

total_losses = [[], []]
total_acc = [[], []]
show_plots = True
show_plots_sigint = False
savemodel = True
earlystop = True
writereport = True
with_face = True
undersampled = False
zero_padding = True
computeclassweight = True

batch_size = 32

########################### System setup ###########################
def handler(signal_received, frame):
    print("\n[System_{}]: SIGINT received".format(time.time()))
    if show_plots_sigint:
        epoch_space = np.arange(1, len(total_losses[1])+1)
        plt.figure()
        plt.plot(epoch_space, total_losses[0])
        plt.plot(epoch_space, total_losses[1])
        plt.legend(["train loss", "validation loss"], loc="lower left")
        plt.savefig('../stats/loss.png')
        plt.close()

        plt.figure()
        plt.plot(epoch_space, total_acc[0])
        plt.plot(epoch_space, total_acc[1])
        plt.legend(["train f1", "validation f1"], loc="lower left")
        plt.savefig('../stats/f1.png')
        plt.close()
        print('Plots saved!')
    print("[System_{}]: Exiting...".format(time.time()))
    sys.exit(0)

signal.signal(signal.SIGINT, handler)
####################################################################

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
device = "cpu"
# torch.backends.cudnn.benchmark = True

# Change the 'train_set_path' according to the location where your 'train_set.csv' is
if undersampled:
    train_set_path = r'../train_dataset_undersampled.csv'
else:
    train_set_path = r'../train_dataset.csv'

# Change the 'landm_csvs_path' according to the directory where your samples of landmarks csvs are
if undersampled and with_face:
    train_landm_csvs_path = r'../points_data/train_pose_face_undersampled'
elif undersampled:
    train_landm_csvs_path = r'../points_data/train_undersampled'
else:
    if with_face:
        train_landm_csvs_path = r'../points_data/train_pose_face'
    else:
        train_landm_csvs_path = r'../points_data/train'

val_set_path = r'../val_dataset.csv'
test_set_path = r'../test_dataset.csv'
if with_face: val_landm_csvs_path = r'../points_data/val_pose_face'
else: val_landm_csvs_path = r'../points_data/val'
test_landm_csvs_path = r'../points_data/test'

def rand_rep(df, max_frames):
    padded = abs(max_frames - df.shape[0])

    while padded != 0:
        rndrow = random.randint(0, len(df)-1)
        rndrepeat = random.randint(1, padded)
        repcol = [df[rndrow].tolist() for _ in range(rndrepeat)]
        df = np.insert(df, rndrow+1, np.array(repcol), axis=0)
        padded -= rndrepeat

    return df

def col_selection(numsel, numpoints, start, end, selist):
    sel = numsel
    sel_flag = True
    for i in range(start, end):
        if sel == 0:
            sel_flag = not sel_flag
            if sel_flag: sel = numsel
            else: sel = numpoints - numsel
        if sel_flag: selist.append(i)
        sel -= 1

class LandmarkDataset(Dataset):
    def __init__(self, tspath, datapath, transform=None):
        self.landmarks_path = datapath
        self.transform = transform
        self.max_len_frame = 10
        self.dataset = pd.read_csv(tspath, header=None)

        # select which columns to keep
        selectedcols = []
        headcols = []
        selectedfromhead = []
        col_selection(2, 4, 0, 92, selectedcols)
        col_selection(2, 3, 92, 155, selectedcols)
        col_selection(2, 3, 155, 218, selectedcols)
        # col_selection(2, 3, 174, 1578, headcols)

        # sel = 2
        # sel_flag = True
        # for x in headcols:
        #     if sel == 0:
        #         sel_flag = not sel_flag
        #         if sel_flag: sel = 2
        #         else: sel = 8
        #     if sel_flag: selectedfromhead.append(x)
        #     sel -= 1
        
        # selectedcols.extend(selectedfromhead)
                
        self.selectedcols = selectedcols

    def __getitem__(self, idx):
        data_path = self.dataset.iloc[idx, 0]
        label = self.dataset.iloc[idx, 1]

        csv_file = os.path.join(self.landmarks_path, data_path.split('/')[0] + "-" + data_path.split('/')[1] + "-" + str(label) + ".csv")
        landmarks = pd.read_csv(csv_file, header=None, dtype=np.float32, usecols=self.selectedcols)

        # equal frames
        landmarks = np.array([landmarks])
        if landmarks.shape[1] < self.max_len_frame:
            if zero_padding:
                landmarks = np.append((landmarks[0]),
                                    [[0] * landmarks.shape[2] for _ in range(abs(self.max_len_frame - landmarks.shape[1]))],
                                    axis=0)
                landmarks = landmarks.astype(np.float32)
            else: landmarks = rand_rep(landmarks[0], self.max_len_frame)
            landmarks = np.array([landmarks])
        elif landmarks.shape[1] > self.max_len_frame:
            landmarks = landmarks[0][:self.max_len_frame]
            landmarks = np.array([landmarks])
        
        sample = {'landmarks': torch.tensor(landmarks).to(device), 'word_label': torch.tensor(label).to(device)}

        if self.transform:
            sample = self.transform(sample)

        return sample['landmarks'], sample['word_label']

    def __len__(self):
        return len(self.dataset)


#Model Architecture
class Network(nn.Module):
    def __init__(self, input_size, seq_len, num_classes, hidden_layer_size, num_layers=1):
        super(Network, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        
        self.cnn = nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=3)

        self.bn = nn.BatchNorm1d(seq_len)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.2)
        self.fdrop = nn.Dropout(p=0.5)

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers,
                            batch_first=True) #When batch_first=True then
                                              #the input and output tensors are provided as (batch, seq, feature)

        self.fc_1 = nn.Linear(hidden_layer_size, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


    def forward(self, input_seq):

        #Initialize hidden and internal state
        h_0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(device)
        c_0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(device)

        #LSTM layer
        lstm_out, (lstm_hn, lstm_cn) = self.lstm(input_seq, (h_0, c_0))
        
        #Reshape LSTM output to fit in Dense layer
        lstm_hn = lstm_hn.view(-1, self.hidden_layer_size)
        out_relu1 = self.dropout(self.relu(lstm_hn))

        #Dense layers
        out_dense1 = self.fc_1(out_relu1)
        # out_relu2 = self.dropout(self.relu(out_dense1))

        return out_dense1

def save_model(epoch, model, optimizer, filename, batch):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'batch': batch
    }

    torch.save(checkpoint, filename)

def load_model(path, model, optim):
    model_name = path.split('/')[-1]
    if os.path.exists(path):
        global batch_size
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['opt'])
        epoch = checkpoint['epoch']
        batch_size = checkpoint['batch']
        print("Model \'{}\' loaded\n".format(model_name))
        return epoch+1
    else:
        print("Model \'{}\' doesn't exist...\n".format(model_name))
        return 0

def arg_to_hot(args, classes_num):
    onehot = []
    for arg in args:
        row = [0] * classes_num
        row[arg] = 1
        onehot.append(row)
    return torch.tensor(onehot).to(device).to(torch.int).numpy()

def plot_grads(params, epoch, batch_num):
    avg_grads = []
    max_grads = []
    layers = []

    for n, p in params:
        if p.requires_grad_ and "bias" not in n:
            layers.append(n)
            avg_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())

    plt.figure()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.6, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), avg_grads, alpha=0.6, lw=1, color="b")
    plt.hlines(0, 0, len(avg_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(avg_grads), 1), layers, rotation=45)
    plt.xlim(left=0, right=len(avg_grads))
    plt.ylim(bottom=-0.001, top=0.02)
    plt.xlabel("Layers")
    plt.ylabel("avg gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig("plots/grads_{}_{}.png".format(epoch, batch_num), bbox_inches='tight', pad_inches=0.3)
    plt.close()

def conf_matr(trues, preds, classes_num, phase, epoch):
    cfmatr = confusion_matrix(trues, preds)
    dfcm = pd.DataFrame(cfmatr/np.max(cfmatr), index = [i for i in range(classes_num)],
                    columns = [i for i in range(classes_num)])
    
    plt.figure(figsize = (12,7))
    sn.heatmap(dfcm, annot=True)
    plt.savefig('../stats/{}/{}_confusion_matrix_{:d}.png'.format(phase, phase, epoch))
    plt.close()

def stats(traint, trainp, valt, valp, classes_num, epoch):
    with open('../stats/{}/{}_report_{:d}.txt'.format('train', 'train', epoch), "w") as f:
        f.write(classification_report(traint, trainp, zero_division=0))
    conf_matr(traint, trainp, classes_num, 'train', epoch)

    with open('../stats/{}/{}_report_{:d}.txt'.format('val', 'val', epoch), "w") as f:
        f.write(classification_report(valt, valp, zero_division=0))
    conf_matr(valt, valp, classes_num, 'val', epoch)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0, verbose=False):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.epoch = -1
        self.model = None
        self.optimizer = None
        self.batch = -1
        self.ytrues = []
        self.ypreds = []
        self.classes_num = -1
        
    def __call__(self, val_loss, epoch, model, optimizer, batch, ytrues, ypreds, classes_num):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.keepmodel(epoch, model, optimizer, batch, ytrues, ypreds, classes_num)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.keepmodel(epoch, model, optimizer, batch, ytrues, ypreds, classes_num)
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.verbose: print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                stats(self.ytrues[0], self.ypreds[0], self.ytrues[1], self.ypreds[1], self.classes_num, self.epoch)
                save_model(self.epoch, self.model, self.optimizer, r'../models/Checkpoints/Checkpoint_{}_{}.pt'.format(self.epoch+1, time.time()), self.batch)
                self.early_stop = True
    
    def keepmodel(self, epoch, model, optimizer, batch, ytrues, ypreds, classes_num):
        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.batch = batch
        self.ytrues = ytrues
        self.ypreds = ypreds
        self.classes_num = classes_num

def train_model(model, loader, criterion, optimizer, early_stopping, classes_num, c_epoch, epochs):
    print('Training begins...\n')

    actions = [model.train, model.eval]
    for epoch in trange(c_epoch, epochs, desc='Epochs'):
        result = ''
        ytrues = [[],[]]
        ypreds = [[],[]]

        for a in range(2):
            actions[a]()
            running_loss = 0.0
            acc = 0.0
            batch_num = 0

            for x, y in loader[a]:
                features = x.to(device)
                features = torch.squeeze(features, 1).to(device)
                labels = y.to(device)
                labels = labels.type(torch.LongTensor).to(device)

                with torch.set_grad_enabled(a == 0):
                    output = model(features) # forward predict
                    y_true = torch.clone(labels).to(device).to(torch.int).numpy()
                    soft = F.softmax(output, dim=1).to(device).to(torch.float32)
                    y_pred = torch.argmax(soft, dim=1).to(device).numpy()
                    ytrues[a].extend(y_true)
                    ypreds[a].extend(y_pred)
                    loss = criterion(output, labels) # loss function compute

                    if a == 0:  # train
                        optimizer.zero_grad() # zero out the gradients
                        loss.backward() # backpropagation
                        # plot_grads(model.named_parameters(), epoch, batch_num)
                        batch_num += 1
                        optimizer.step() # update weights
                    
                    running_loss += loss.item() * features.size(0)
                    acc += f1_score(y_true, y_pred, average='macro', zero_division=0) * features.size(0)

            epoch_loss = running_loss / len(loader[a].dataset)
            epoch_acc = acc / len(loader[a].dataset)
            total_losses[a].append(epoch_loss)
            total_acc[a].append(epoch_acc)

            epoch_action = "Train" if a == 0 else "Validation"
            result += "{} loss: {:.4f} -- {} score: {:.4f}\n".format(epoch_action, epoch_loss, epoch_action, epoch_acc)

            if a == 1:
                if earlystop:
                    if not early_stopping.early_stop: 
                        early_stopping(epoch_loss, epoch, model, optimizer, batch_size, ytrues, ypreds, classes_num)
        
        print(result)
        if writereport:
            if epoch + 1 == epochs and early_stopping.early_stop == False:
                stats(ytrues[0], ypreds[0], ytrues[1], ypreds[1], classes_num, epoch)

        if savemodel:
            try:
                if epoch + 1 == epochs and early_stopping.early_stop == False:
                    save_model(epoch, model, optimizer, r'../models/Checkpoints/Checkpoint_{}_{}.pt'.format(epoch+1, time.time()), batch_size)
            except:
                print('Error: Model could not be saved!')
        
    print('Finished training!\n')
    return model, total_losses[0], total_losses[1], total_acc[0], total_acc[1]


def main():
    epochs = 150
    classes_num = 10
    lr = 0.0001 # learning rate

    c_epoch = 0

    print("Train landmark dataset: {}".format(train_landm_csvs_path))
    print("Validation landmark dataset: {}".format(val_landm_csvs_path))
    
    train = LandmarkDataset(train_set_path, train_landm_csvs_path)
    val = LandmarkDataset(val_set_path, val_landm_csvs_path)
    # mp.set_start_method('spawn', force=True)
    loader = [DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4), DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=4)]

    input_size = train[0][0].size(2)
    seq_len = train[0][0].size(1) # sequence leangth for LSTM
    hidden_size = 32 # number of hidden cells
    layers_num = 1 # stacked LSTM layers

    class_weight = None
    if computeclassweight:
        y = train.dataset[1].to_numpy()
        class_weight = compute_class_weight(class_weight='balanced', classes=np.array(list(range(classes_num))), y=y)
        class_weight = torch.tensor(class_weight, dtype=torch.float32)

    model = Network(input_size, seq_len, classes_num, hidden_size, layers_num).to(device) # initialize the model
    criterion = None
    if computeclassweight: criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    else: criterion = torch.nn.CrossEntropyLoss() # cross entropy loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001) # Adam optimizer

    early_stopping = EarlyStopping(patience=7)

    lstm, train_loss, val_loss, train_f1, val_f1 = train_model(model, loader, criterion, optimizer, early_stopping, classes_num, c_epoch, epochs)

    if show_plots:
        epoch_space = np.arange(1, len(val_loss)+1)
        ymin = min(min(train_loss), min(val_loss))
        ymax = max(max(train_loss), max(val_loss))
        legendlist = []
        plt.figure()
        plt.plot(epoch_space, train_loss)
        plt.plot(epoch_space, val_loss)
        if early_stopping.early_stop:
            plt.vlines(float(early_stopping.epoch+1), ymin, ymax, colors='r', linestyles='dashed')
            legendlist.extend(["train loss", "validation loss", "early stopping checkpoint"])
        else: legendlist.extend(["train loss", "validation loss"])
        plt.legend(legendlist, loc="upper right")
        plt.savefig('../stats/loss.png')
        plt.close()

        ymin = min(min(train_f1), min(val_f1))
        ymax = max(max(train_f1), max(val_f1))
        legendlist.clear()
        legendlist = []
        plt.figure()
        plt.plot(epoch_space, train_f1)
        plt.plot(epoch_space, val_f1)
        if early_stopping.early_stop:
            plt.vlines(float(early_stopping.epoch+1), ymin, ymax, colors='r', linestyles='dashed')
            legendlist.extend(["train f1", "validation f1", "early stopping checkpoint"])
        else: legendlist.extend(["train f1", "validation f1"])
        plt.legend(legendlist, loc="lower right")
        plt.savefig('../stats/f1.png')
        plt.close()

if __name__ == '__main__':
    main()



