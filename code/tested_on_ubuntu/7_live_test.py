import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import signal
import sys
import torch.nn as nn
from torchvision import transforms
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
import mediapipe as mp
import cv2

# torch.autograd.set_detect_anomaly(True)

HAND_POINTS = 21

train_set_path = ''
val_set_path = ''
test_set_path = ''
train_landm_csvs_path = ''
val_landm_csvs_path = ''
test_landm_csvs_path = ''
root_video_path = r'../GSL_isol'

total_losses = []
total_acc = []
show_plots = False
show_plots_sigint = False
writereport = False
with_face = True
undersampled = False
zero_padding = True
computeclassweight = True

labels = ['ESU','EGW(1)', 'KALO', 'SYN', 'BIBLIO', 'XREIAZOMAI', 'EXW', 'GIA', 'ENTAKSEI', 'GEIA']

batch_size = 64

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

if undersampled:
    train_set_path = r'../train_dataset_undersampled.csv'
else:
    train_set_path = r'../train_dataset.csv'

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
test_landm_csvs_path = r'../points_data/test_pose_face'

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
        col_selection(2, 4, 0, 92, selectedcols)
        col_selection(2, 3, 92, 155, selectedcols)
        col_selection(2, 3, 155, 218, selectedcols)
                
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
        # self.fdrop = nn.Dropout(p=0.5)

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers,
                            batch_first=True) #When batch_first=True then
                                              #the input and output tensors are provided as (batch, seq, feature)

        self.fc_1 = nn.Linear(hidden_layer_size, num_classes)
        # self.fc_output = nn.Linear(128, num_classes)
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
    plt.savefig('../stats/{}_confusion_matrix_{:d}.png'.format(phase, epoch))
    plt.close()

def stats(testt, testp, classes_num, epoch):
    with open('../stats/{}_report_{:d}.txt'.format('train', epoch), "w") as f:
        f.write(classification_report(testt, testp, zero_division=0))
    conf_matr(testt, testp, classes_num, 'test', epoch)

def test_model(model, features):
    ypred = None
    model.eval()
    prob = 0.0

    features = features.to(device)
    # features = torch.squeeze(features, 1).to(device)

    with torch.set_grad_enabled(False):
        output = model(features)
        soft = F.softmax(output, dim=1).to(device).to(torch.float32)
        ypred = torch.argmax(soft, dim=1).to(device).numpy()
        prob = soft[0][ypred[0]]
    
    return ypred[0], prob

class ImgAnnotator():
    def __init__(self, device):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.device = device
        self.points_range = range(23)
        self.points_with_face = 23*2
    
    def capture_video(self):
        featurelist = []
        framelist = []

        classes_num = 10
        lr = 0.0001

        input_size = 130
        seq_len = 10
        hidden_size = 32
        layers_num = 1

        model = Network(input_size, seq_len, classes_num, hidden_size, layers_num).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        load_model('./Checkpoint_117_1624070170.6162398.pt', model, optimizer)

        cap = cv2.VideoCapture(0)
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

            frame = 0
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = holistic.process(image)

                if results.pose_landmarks != None:
                    for point in self.points_range:
                        featurelist.append(str(results.pose_landmarks.landmark[point].x))
                        featurelist.append(str(results.pose_landmarks.landmark[point].y))
                        # featurelist.append(str(results.pose_landmarks.landmark[point].z))
                        # featurelist.append(str(results.pose_landmarks.landmark[point].visibility))
                else: featurelist.extend(['0' for _ in range(self.points_with_face)])

                if(results.left_hand_landmarks != None):
                    for point in range(HAND_POINTS):
                        featurelist.append(str(results.left_hand_landmarks.landmark[point].x))
                        featurelist.append(str(results.left_hand_landmarks.landmark[point].y))
                        # featurelist.append(str(results.left_hand_landmarks.landmark[point].z))
                else: featurelist.extend(['0' for _ in range(HAND_POINTS*2)])

                if(results.right_hand_landmarks != None):
                    for point in range(HAND_POINTS):
                        featurelist.append(str(results.right_hand_landmarks.landmark[point].x))
                        featurelist.append(str(results.right_hand_landmarks.landmark[point].y))
                        # featurelist.append(str(results.right_hand_landmarks.landmark[point].z))
                else: featurelist.extend(['0' for _ in range(HAND_POINTS*2)])

                if frame < 10:
                    framelist.append(featurelist.copy())
                    frame += 1
                else:
                    framelist.pop(0)
                    framelist.append(featurelist.copy())
                
                featurelist.clear()
                
                if frame == 10:
                    # predict
                    features = np.array([framelist], dtype=np.float32)
                    feat_torch = torch.from_numpy(features)

                    # print on image
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    textPos = (20,50)
                    probPos = (20, 100)
                    fontScale = 1
                    fontColor = (255,255,255)
                    lineType = 2

                    ypred, prob = test_model(model, feat_torch)

                    cv2.putText(image,'Class: {}'.format(labels[ypred]), 
                                textPos, 
                                font, 
                                fontScale,
                                fontColor,
                                lineType)

                    cv2.putText(image,'Prob: {:.2f}%'.format(prob*100.0), 
                                probPos, 
                                font, 
                                fontScale,
                                fontColor,
                                lineType)

                # Draw landmark annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.mp_drawing.draw_landmarks(
                    image, results.face_landmarks, self.mp_holistic.FACE_CONNECTIONS)
                self.mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
                cv2.imshow('MediaPipe Holistic', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()


def main():
    annot = ImgAnnotator(device)
    annot.capture_video()

if __name__ == '__main__':
    main()