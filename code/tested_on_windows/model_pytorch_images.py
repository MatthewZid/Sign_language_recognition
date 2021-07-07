import pandas as pd
import torch
import numpy as np
import cv2
import os
import io
import shutil

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchinfo import summary

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score

import seaborn as sn
import matplotlib.pyplot as plt


class LandmarksDataset(Dataset):

    def __init__(self, csv_words_paths, landmarks_csvs_dir, transform=None):
        """

        :param csv_words_paths: Path to the csv file with the target words and their landmarks paths
        :param landmarks_csvs_dir: Directory with all the csvs of landmarks
        :param transform: Transform to be applied
        """
        self.words_paths = pd.read_csv(csv_words_paths)
        self.landmarks_dir = landmarks_csvs_dir
        self.transform = transform
        self.max_len_frame = int((self.landmarks_dir.split('\\'))[-2].split('_')[-1])
        self.individual_words = self.words_paths.columns
        self.individual_words = self.individual_words.to_list()
        self.words_to_numbers = np.arange(
            len(self.individual_words)).tolist()  # The representation of each word as a number is just its index in individual_words list
        count_samples = []
        for col in self.words_paths.columns:
            count_samples.append(self.words_paths[col].count())

        # One type oh weighting
        # self.sample_weights = [1 - (x_sample / sum(count_samples)) for x_sample in count_samples]

        # Another type of weighting
        weights = count_samples / sum(count_samples)
        weights = 1.0 / weights
        self.sample_weights = weights / sum(weights)

    def __len__(self):
        overall_samples = np.sum(self.words_paths.count())
        return overall_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Each column does not have equal samples (there are NaN values)
        # So, let calculate the cumulative length column by column in order to return the sample with
        # the cumulative 'idx'
        cumulative_column_length = 0
        for column in self.words_paths.columns:
            column_idx = 0
            column_length = self.words_paths[column].count()
            cumulative_column_length += column_length
            if idx < cumulative_column_length:
                column_idx = idx - (cumulative_column_length - column_length)
                break

        word_path_name = self.words_paths[column].iloc[column_idx]
        word_path = os.path.join(self.landmarks_dir, word_path_name)

        frames = os.listdir(word_path)
        ind = 0
        for frame in frames:
            if ind == 0:

                img_path = os.path.join(word_path, frame)
                temp = cv2.imdecode(np.fromfile(u'' + img_path, np.uint8), cv2.IMREAD_UNCHANGED)
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
                img_height = temp.shape[0]
                img_width = temp.shape[1]
                img_channels = temp.shape[2]
                if img_height > img_width:
                    dif = img_height - img_width
                    temp = temp[int(dif/2):-int(dif/2), :]
                elif img_height < img_width:
                    dif = img_width - img_height
                    temp = temp[:, int(dif / 2):-int(dif / 2)]
                temp = cv2.resize(temp, (128, 128), interpolation=cv2.INTER_AREA)
                img_height = temp.shape[0]
                img_width = temp.shape[1]

                stacked_imgs = np.expand_dims(temp, axis=0)
                stacked_imgs = stacked_imgs.astype('float64')
                stacked_imgs *= 1.0/255.0


            elif ind < self.max_len_frame:
                img_path = os.path.join(word_path, frame)
                img = cv2.imdecode(np.fromfile(u'' + img_path, np.uint8), cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_height = img.shape[0]
                img_width = img.shape[1]
                if img_height > img_width:
                    dif = img_height - img_width
                    img = img[int(dif/2):-int(dif/2), :]
                elif img_height < img_width:
                    dif = img_width - img_height
                    img = img[:, int(dif / 2):-int(dif / 2)]
                img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                img_height = img.shape[0]
                img_width = img.shape[1]

                img = img.astype('float64')
                img *= 1.0 / 255.0
                temp = np.vstack((stacked_imgs, np.expand_dims(img, axis=0)))
                stacked_imgs = temp.copy()

            else:
                break

            ind += 1


        stacked_imgs = np.array([stacked_imgs])

        if abs(self.max_len_frame - stacked_imgs.shape[1]) != 0:

            stacked_imgs = np.append((stacked_imgs[0]), np.zeros((self.max_len_frame - stacked_imgs.shape[1], img_height, img_width, img_channels), dtype=np.uint8), axis=0)
            stacked_imgs = np.array([stacked_imgs])
            stacked_imgs = stacked_imgs.astype('float64')



        word_label = column
        word_index = self.individual_words.index(word_label)
        word_number_label = self.words_to_numbers[word_index]

        stacked_imgs = stacked_imgs.reshape(1, self.max_len_frame, img_channels, img_height, img_width)
        sample = {'landmarks': stacked_imgs, 'word_label': word_number_label}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in samples to Tensors"""

    def __call__(self, sample):
        landmarks, word_label = sample['landmarks'], sample['word_label']

        return {'landmarks': torch.from_numpy(landmarks), 'word_label': word_label}


# Model Architecture
class Network(nn.Module):
    def __init__(self, num_chans, image_heig, image_wid, num_classes, hidden_layer_size, num_layers=1, num_filters=64, kernel_size=5,
                 dilation=1, stride=1,
                 padding=0):
        super().__init__()

        self.num_classes = num_classes

        #### 1D CNN ###
        self.cnn_flag = True

        if self.cnn_flag:
            # 1st 2DConv layer
            conv1_num_filters = 16
            conv1_kernel_size = kernel_size
            conv1_dilation = dilation
            conv1_padding = padding
            conv1_stride = stride
            self.conv1 = nn.Conv2d(in_channels=num_chans, out_channels=conv1_num_filters*num_chans, kernel_size=conv1_kernel_size,
                                   dilation=conv1_dilation, padding=conv1_padding,
                                   stride=conv1_stride)

            # 1st 2DMaxPooling layer
            maxpool1_kernel_size = kernel_size
            maxpool1_dilation = dilation
            maxpool1_padding = padding
            maxpool1_stride = maxpool1_kernel_size  # Default for stride of maxPool2D is stride=kernel_size
            self.maxPool1 = nn.MaxPool2d(kernel_size=maxpool1_kernel_size, dilation=maxpool1_dilation,
                                         padding=maxpool1_padding, stride=maxpool1_stride)

            # 2nd 1DConv layer
            # conv2_num_filters = 64
            # conv2_kernel_size = kernel_size
            # conv2_dilation = dilation
            # conv2_padding = padding
            # conv2_stride = stride
            # self.conv2 = nn.Conv1d(in_channels=conv1_num_filters, out_channels=conv2_num_filters, kernel_size=conv2_kernel_size,
            #                        dilation=conv2_dilation, padding=conv2_padding,
            #                        stride=conv1_stride)

            # 2nd 1DMaxPooling layer
            # maxpool2_kernel_size = kernel_size
            # maxpool2_dilation = dilation
            # maxpool2_padding = padding
            # maxpool2_stride = maxpool2_kernel_size  # Default for stride of maxPool1D is stride=kernel_size
            # self.maxPool2 = nn.MaxPool1d(kernel_size=maxpool2_kernel_size, dilation=maxpool2_dilation,
            #                              padding=maxpool2_padding, stride=maxpool2_stride)

            # Calculate cnn output size
            conv1_out_h_out_size = int(((image_heig + (2 * conv1_padding) - (conv1_dilation * (
                        conv1_kernel_size - 1)) - 1) / conv1_stride) + 1)  # Where h_out is H_out of pytorch documentation for conv2D
            conv1_out_w_out_size = int(((image_wid + (2 * conv1_padding) - (conv1_dilation * (
                    conv1_kernel_size - 1)) - 1) / conv1_stride) + 1)  # Where w_out is W_out of pytorch documentation for conv2D
            conv1_out_channels_size = conv1_num_filters * num_chans

            maxpool1_out_h_out_size = int(((conv1_out_h_out_size + (2 * maxpool1_padding) - (
                        maxpool1_dilation * (maxpool1_kernel_size - 1)) - 1) / maxpool1_stride) + 1)
            maxpool1_out_w_out_size = int(((conv1_out_w_out_size + (2 * maxpool1_padding) - (
                    maxpool1_dilation * (maxpool1_kernel_size - 1)) - 1) / maxpool1_stride) + 1)

            # conv2_out_l_size = int(((maxpool1_out_l_size + (2 * conv2_padding) - (
            #             conv2_dilation * (conv2_kernel_size - 1)) - 1) / conv2_stride) + 1)
            # maxpool2_out_l_size = int(((conv2_out_l_size + (2*maxpool2_padding) - (maxpool2_dilation*(maxpool2_kernel_size-1))-1)/maxpool2_stride)+1)

            input_size_for_next_layer = maxpool1_out_h_out_size * maxpool1_out_w_out_size * conv1_out_channels_size  # maxpool2_out_l_size*conv2_num_filters

        ###LSTM OR GRU####
        self.num_layers = num_layers

        self.hidden_layer_size = hidden_layer_size

        if self.cnn_flag:
            inp_sz = input_size_for_next_layer
            # self.dropout_cnn_1 = nn.Dropout(0.2)
            self.final_dropout_cnn = nn.Dropout(0.2)

        else:
            inp_sz = input_size

        ###LSTM####
        self.lstm_flag = False

        if self.lstm_flag:
            self.lstm = nn.LSTM(inp_sz, hidden_layer_size, num_layers,
                                batch_first=True)  # When batch_first=True then
            # the input and output tensors are provided as (batch, seq, features)
            self.final_lstm_dropout = nn.Dropout(0.2)

        ###GRU####
        self.gru_flag = True

        if self.gru_flag:
            self.gru = nn.GRU(inp_sz, hidden_layer_size, num_layers,
                              batch_first=True)  # When batch_first=True then
            # the input and output tensors are provided as (batch, seq, features)

            self.final_gru_dropout = nn.Dropout(0.2)

        # First Dense
        self.fc_1 = nn.Linear(hidden_layer_size, 128)

        # First Dropout
        self.dropout_1 = nn.Dropout(0.2)

        # Second Dense
        # self.fc_2 = nn.Linear(128, 64)

        # Second Dropout
        # self.dropout_2 = nn.Dropout(0.2)

        # Final Dense
        self.fc_output = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

        # self.softmax = nn.Softmax()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):

        if self.cnn_flag:
            #####CNN 1D######
            size_input = list(input_seq.size())

            cnn_input_seq = input_seq.double()
            cnn_input_seq = cnn_input_seq.view(size_input[0] * size_input[1],
                                               size_input[2], size_input[3],
                                               size_input[4])  # Flat first dim (batches) with second dim (seq)

            # 1st cnn layer
            conv1_output = self.conv1(cnn_input_seq)
            maxpool1_output = self.maxPool1(conv1_output)
            out_relu_maxpool1 = self.relu(maxpool1_output)

            # First dropout for cnn layers
            # cnn_dropout1 = self.dropout_cnn_1(out_relu_maxpool1)

            # 2nd cnn layer
            # conv2_output = self.conv2(cnn_dropout1)
            # maxpool2_output = self.maxPool1(conv2_output)
            # out_relu_maxpool2 = self.relu(maxpool2_output)

            # Final cnn output
            cnn_output_seq = out_relu_maxpool1 # out_relu_maxpool2

            size_cnn_output_seq = list(cnn_output_seq.size())
            first_reshape_cnn_output_seq = cnn_output_seq.view(size_cnn_output_seq[0],
                                                               size_cnn_output_seq[1] * size_cnn_output_seq[2] * size_cnn_output_seq[3])
            second_reshape_cnn_output_seq = first_reshape_cnn_output_seq.view(size_input[0], size_input[1],
                                                                              size_cnn_output_seq[1] *
                                                                              size_cnn_output_seq[2] * size_cnn_output_seq[3])

            # Dropout of final cnn layer
            final_dropout_cnn = self.final_dropout_cnn(second_reshape_cnn_output_seq)
            cnn_output_for_next_layer_input = final_dropout_cnn  # cnn_output_for_next_layer_input = second_reshape_cnn_output_seq

        ########LSTM######
        if self.lstm_flag:
            if self.cnn_flag:
                lstm_in = cnn_output_for_next_layer_input
                lstm_in = lstm_in.double()
            else:
                lstm_in = input_seq.type(torch.FloatTensor)  # Transform to FloatTensor for better performance in gpu

            # Initialize hidden and internal state

            h_0 = Variable(torch.zeros(self.num_layers, lstm_in.size(0), self.hidden_layer_size))
            c_0 = Variable(torch.zeros(self.num_layers, lstm_in.size(0), self.hidden_layer_size))
            if self.cnn_flag:
                h_0 = h_0.double()
                c_0 = c_0.double()

            # LSTM layer

            lstm_out, (lstm_hn, lstm_cn) = self.lstm(lstm_in, (h_0, c_0))

            # Reshape LSTM last hidden layer to fit in Dense layer
            lstm_hn = lstm_hn[-1].view(-1, self.hidden_layer_size)
            out_relu1 = self.relu(lstm_hn)

            # Final LSTM dropout
            # final_lstm_drop = self.final_lstm_dropout(out_relu1)

            # Final LSTM output
            final_output = out_relu1  # final_lstm_drop

        #####GRU#####
        if self.gru_flag:
            if self.cnn_flag:
                gru_in = cnn_output_for_next_layer_input
                gru_in = gru_in.double()
            else:
                gru_in = input_seq.type(torch.FloatTensor)  # Transform to FloatTensor for better performance in gpu

            # Initialize hidden and internal state

            h_0 = Variable(torch.zeros(self.num_layers, gru_in.size(0), self.hidden_layer_size))

            if self.cnn_flag:
                h_0 = h_0.double()

            # GRU layer

            gru_out, gru_hn = self.gru(gru_in, h_0)

            # Reshape LSTM last hidden layer to fit in Dense layer
            gru_hn = gru_hn[-1].view(-1, self.hidden_layer_size)
            out_relu1 = self.relu(gru_hn)

            # Final GRU dropout
            final_gru_drop = self.final_gru_dropout(out_relu1)

            # Final GRU output
            final_output = final_gru_drop  # out_relu1

        ##### Dense layers #######

        # First Dense
        out_dense1 = self.fc_1(final_output)
        out_relu2 = self.relu(out_dense1)

        # First Dropout
        out_dropout1 = self.dropout_1(out_relu2)

        # Second Dense
        # out_dense2 = self.fc_2(out_relu2)
        # out_relu3 = self.relu(out_dense2)

        # Second Dropout
        # out_dropout2 = self.dropout_2(out_relu3)

        # Final Dense
        out_dense_output = self.fc_output(out_dropout1)

        # out_sigmoid = self.sigmoid(out_dense2)
        # out_softmax = self.softmax(out_dense2) #It isnt needed for MultiLabelSoftMarginLoss() nor for CrossEntropy

        return out_dense_output  # out_dense2  # out_sigmoid  # out_softmax


def loss_batch(modell, loss_func, xb, yb, words_to_nums, opt=None):
    predictions = modell(xb)

    if isinstance(criterion, type(nn.MultiLabelSoftMarginLoss())):

        y_transformed = one_hot_transform_labels(yb, words_to_nums)
        losss = loss_func(predictions, y_transformed)

    elif isinstance(criterion, type(nn.CrossEntropyLoss())):

        losss = loss_func(predictions, yb)

    if opt is not None:
        opt.zero_grad()  # Reset gradients

        losss.backward()  # Calculates gradients

        opt.step()  # Update parameters

    return losss.item(), len(xb), predictions


def evaluate(modell, loss_fn, vl_dg, words_to_nums):
    with torch.no_grad():
        # Pass each batch through the model
        yb_all = []
        losses_all = []
        nums_all = []
        predictions_all = []
        for batch_smpl in vl_dg:
            yb = (torch.from_numpy(np.asarray(batch_smpl['word_label']))).to(device)
            yb_all.append(yb)
            results = [loss_batch(modell, loss_fn,
                                  torch.squeeze((torch.from_numpy(np.asarray(batch_smpl['landmarks']))).to(device), 1),
                                  yb,
                                  words_to_nums
                                  )
                       ]

            # Separate losses, counts, predictions
            losses, nums, predictions = zip(*results)

            losses_all.append(losses)
            nums_all.append(nums)
            predictions_all.append(predictions)

        # Total size of dataset
        total = np.sum(nums_all)

        # Avg loss across batches
        avg_loss = np.sum(np.multiply(losses_all, nums_all)) / total

        return avg_loss, predictions_all, yb_all


def one_hot_transform_labels(labels, words_to_number):
    one_hot_labels = []

    if torch.is_tensor(labels):

        batch_sz = list(labels.size())[0]

    else:

        batch_sz = labels.shape[0]

    for batch in range(batch_sz):
        zeros = [0] * len(words_to_number)

        for word_idx in range(len(words_to_number)):
            if labels[batch] == words_to_number[word_idx]:
                zeros[word_idx] = 1
                one_hot_labels.append(zeros)
                break

    if torch.is_tensor(labels):
        one_hot_labels = (torch.from_numpy(np.array(one_hot_labels))).type(torch.LongTensor)

    else:
        one_hot_labels = np.array(one_hot_labels)

    return one_hot_labels


def classific_report(y_true, y_pred, probabilities, indiv_words, words_to_num, epch, epchs, earl_stop, pth_results,
                     pth_txt, dataset_type="val"):
    cf_matrix = confusion_matrix(y_true, y_pred, labels=range(len(indiv_words)))

    prnt_out = '\nConfusion matrix:\n'
    print(prnt_out)
    print(cf_matrix)
    with open(pth_txt, 'a') as ff:
        ff.write(prnt_out)
        ff.write(np.array2string(cf_matrix))

    clas_report = classification_report(y_true, y_pred, target_names=indiv_words, labels=range(len(indiv_words)),
                                        zero_division=0)
    prnt_out = '\nClassification report:\n'
    print(prnt_out)
    print(clas_report)
    with io.open(pth_txt, 'a', encoding="utf-8") as ff:
        ff.write(prnt_out)
        ff.write(clas_report)

    f1_macro = f1_score(y_true, y_pred, average='macro')
    prnt_out = '\nF1 macro\n'
    print(prnt_out)
    print(f1_macro)
    with open(pth_txt, 'a') as ff:
        ff.write(prnt_out)
        ff.write(str(f1_macro))

    #############IMPORTANT##############
    # One-hot representation is needed both for roc-auc scores and roc-auc curve
    y_transformed = one_hot_transform_labels(y_true, words_to_num)

    prnt_out = '\nROC-AUC score:\n'
    print(prnt_out)
    try:
        roc_auc_scrs = roc_auc_score(y_transformed, probabilities)
    except ValueError:
        roc_auc_scrs = 0  # If only one class has predicted values

    print(roc_auc_scrs)
    with open(pth_txt, 'a') as ff:
        ff.write(prnt_out)
        ff.write(str(roc_auc_scrs))

    if epch == epchs or earl_stop:

        # Plot confusion matrix
        df_cm = pd.DataFrame(cf_matrix, index=indiv_words, columns=indiv_words)
        plt.figure(figsize=(12, 10))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(pth_results + "/Confusion_matrix_" + dataset_type + ".png")
        plt.show()

        # Plot ROC-AUC curve
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(indiv_words)):
            fpr[i], tpr[i], _ = roc_curve(y_transformed[:, i], probabilities[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig1, ax2 = plt.subplots(1, 1, figsize=(25, 10))
        for i, val in enumerate(indiv_words):
            ax2.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area{1:0.2f})'.format(val, roc_auc[i]))

        ax2.plot([0, 1], [0, 1], 'k--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic to multi-class')
        plt.legend(loc='lower right')
        plt.savefig(pth_results + "/ROC-AUC_" + dataset_type + ".png")
        plt.show()

    return roc_auc_scrs, f1_macro


def create_losses_plot(epchs, train_losses, val_losses, crit_to_str, pth_results):
    fig, ax1 = plt.subplots(1, 1, figsize=(25, 10))

    epochs_list = np.arange(1, epchs + 1)
    ax1.plot(epochs_list, train_losses, color='b', label='Loss')
    ax1.plot(epochs_list, val_losses, color='orange', label='Val_loss')
    ax1.set_xticks(range(1, epchs + 1))
    ax1.set_title('Training Error')
    ax1.set_ylabel(crit_to_str, fontsize=12)
    ax1.set_xlabel('#Epoch', fontsize=12)
    plt.legend()
    plt.savefig(pth_results + "/Losses_" + crit_to_str + '.png')
    # plt.close(fig)
    plt.show()


def create_roc_auc_scores_plot(epchs, roc_auc_scores, pth_results):
    fig, ax3 = plt.subplots(1, 1, figsize=(25, 10))

    epochs_list = np.arange(1, epchs + 1)
    ax3.plot(epochs_list, roc_auc_scores)
    ax3.set_xticks(range(1, epchs + 1))
    ax3.set_title('ROC-AUC scores plot')
    ax3.set_ylabel('ROC-AUC score', fontsize=12)
    ax3.set_xlabel('#Epoch', fontsize=12)
    plt.savefig(pth_results + "/ROC-AUC-scores.png")
    # plt.close(fig)
    plt.show()


def create_f1_macros_plot(epchs, f1_macros, pth_results):
    fig, ax3 = plt.subplots(1, 1, figsize=(25, 10))

    epochs_list = np.arange(1, epchs + 1)
    ax3.plot(epochs_list, f1_macros)
    ax3.set_xticks(range(1, epchs + 1))
    ax3.set_title('F1 macro scores plot')
    ax3.set_ylabel('F1 macro', fontsize=12)
    ax3.set_xlabel('#Epoch', fontsize=12)
    plt.savefig(pth_results + "/f1_macro-scores.png")
    # plt.close(fig)
    plt.show()


if __name__ == '__main__':

    # Change the 'train_set_path' according to the location where your 'train_set.csv' is
    train_set_path = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\datasets\univ_patras\GSL_split\GSL_split\GSL_isolated\train_set.csv'

    # Change the 'landm_csvs_path' according to the directory where your samples of landmarks csvs are
    train_landm_csvs_path = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\datasets\univ_patras\GSL_split\GSL_split\GSL_isolated\annotated_imgs_frames_10\train'

    # Create a pytorch training Dataset
    train_landmarks_dataset = LandmarksDataset(csv_words_paths=train_set_path, landmarks_csvs_dir=train_landm_csvs_path,
                                               transform=transforms.Compose([ToTensor()]))

    # Change the 'val_set_path' according to the location where your 'val_set.csv' is
    val_set_path = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\datasets\univ_patras\GSL_split\GSL_split\GSL_isolated\val_set.csv'

    # Change the 'val_landm_csvs_path' according to the directory where your validation samples of landmarks csvs are
    val_landm_csvs_path = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\datasets\univ_patras\GSL_split\GSL_split\GSL_isolated\annotated_imgs_frames_10\val'

    # Create a pytorch validation Dataset
    val_landmarks_dataset = LandmarksDataset(csv_words_paths=val_set_path, landmarks_csvs_dir=val_landm_csvs_path,
                                             transform=transforms.Compose([ToTensor()]))

    # Change the 'test_set_path' according to the location where your 'test_set.csv' is
    test_set_path = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\datasets\univ_patras\GSL_split\GSL_split\GSL_isolated\test_set.csv'

    # Change the 'test_landm_csvs_path' according to the directory where your validation samples of landmarks csvs are
    test_landm_csvs_path = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\datasets\univ_patras\GSL_split\GSL_split\GSL_isolated\annotated_imgs_frames_10\test'

    # Create a pytorch testing Dataset
    test_landmarks_dataset = LandmarksDataset(csv_words_paths=test_set_path, landmarks_csvs_dir=test_landm_csvs_path,
                                              transform=transforms.Compose([ToTensor()]))

    words_to_numbers = train_landmarks_dataset.words_to_numbers
    individual_words = train_landmarks_dataset.individual_words

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Define parameters for the pythorch DataLoader
    batch_size = 64
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 8}

    # Create pytorch DataLoader
    training_generator = DataLoader(train_landmarks_dataset, **params)
    validation_generator = DataLoader(val_landmarks_dataset, **params)
    testing_generator = DataLoader(test_landmarks_dataset, **params)

    # Define parameters
    epochs = 500

    # Get input size (num_features) and num_classes (num_words)
    for batch_sample in train_landmarks_dataset:
        num_channels = list(batch_sample['landmarks'].size())[2]
        image_height = list(batch_sample['landmarks'].size())[3]
        image_width = list(batch_sample['landmarks'].size())[4]
        window_size = list(batch_sample['landmarks'].size())[1]

        break

    learning_rate = 0.001
    hidden_size = 20
    nm_layers = 1
    nm_classes = len(train_landmarks_dataset.individual_words)

    # Instantiate Network
    model = Network(num_channels, image_height, image_width, nm_classes, hidden_size, nm_layers)
    if model.cnn_flag:
        model = model.double()

    # Criterion
    criterion = nn.CrossEntropyLoss()  # nn.MultiLabelSoftMarginLoss()
    criterion_to_str = 'CrossEntropyLoss'  # 'MultiLabelSoftMarginLoss'

    if criterion_to_str == 'CrossEntropyLoss':
        weighting_flag = False

        # Define weight for weighted CrossEntropy loss
        if weighting_flag:
            if model.cnn_flag:
                ws = torch.DoubleTensor(train_landmarks_dataset.sample_weights).to(device)

            else:
                ws = torch.FloatTensor(train_landmarks_dataset.sample_weights).to(device)

            criterion = nn.CrossEntropyLoss(weight=ws)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # , weight_decay=0.0001
    optimizer_to_str = 'Adam'

    # Path to save model for each epoch
    path_save_model = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\temp_saved_models'
    if not os.path.exists(path_save_model):
        os.makedirs(path_save_model)

    # Define early stopping parameters
    patience = 10
    epochs_no_improve = 0
    early_stop = False
    min_delta = 0.0001
    min_val_loss = np.Inf  # Early stopping applied in validation loss to prevent overfitting

    # Path to save results
    path_results = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\logs'
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    # Save model architecture
    model_summary = summary(model, input_size=(
        batch_size, list(batch_sample['landmarks'].size())[1], list(batch_sample['landmarks'].size())[2],
        list(batch_sample['landmarks'].size())[3], list(batch_sample['landmarks'].size())[4]))

    path_model_arch = path_results + '/model_architecture.txt'
    with io.open(path_model_arch, 'w', encoding="utf-8") as f:
        f.write(str(model_summary))

    # Txt file path for results of each epoch
    path_txt = path_results + '/epochs_results.txt'

    # Save parameters definition
    param_defs = 'Using: Criterion=' + criterion_to_str + ' Optimizer=' + optimizer_to_str + ' with learning rate=' + str(
        learning_rate) + '\n'
    with open(path_txt, 'w') as f:
        f.write(param_defs)

    epoch_losses = []
    epoch_nums = []

    epoch_val_losses = []
    avg_train_losses = []

    val_auc_rocs = []
    f1_macrs = []

    for epoch in range(epochs):

        # Training
        model.train()

        batch_losses = []
        num = []
        batc = 0
        for batch_sample in training_generator:
            # Transfer to GPU or CPU
            batch_sample_landmarks = torch.from_numpy(np.asarray(batch_sample['landmarks']))
            batch_sample_word_label = torch.from_numpy(np.asarray(batch_sample['word_label']))

            x, y = batch_sample_landmarks.to(device), batch_sample_word_label.to(device)

            x = torch.squeeze(x, 1)

            num.append(len(x))

            loss, _, preds = loss_batch(model, criterion, x, y, words_to_numbers, optimizer)

            batch_losses.append(loss)

            print('Current batch is: ' + str(batc+1))
            batc += 1

        epoch_losses.append(batch_losses)  # Store the loss of each batch
        epoch_nums.append(num)  # Store the number of losses

        avg_train_loss = np.sum(np.multiply(batch_losses, num)) / np.sum(num)  # Calculate the average of batch losses
        avg_train_losses.append(avg_train_loss)  # Store the average loss

        # Evaluation
        model.eval()
        avg_val_loss, preds_eval, yb_eval_all_true = evaluate(model, criterion, validation_generator, words_to_numbers)
        epoch_val_losses.append(avg_val_loss)

        # Check the difference of min validation loss and current validation loss
        if avg_val_loss < min_val_loss:

            # Save the model temporarily
            str_epoch = epoch + 1
            if not os.path.exists(path_save_model + '\epoch_' + str(str_epoch)):
                os.makedirs(path_save_model + '\epoch_' + str(str_epoch))

            torch.save(model.state_dict(model.state_dict()),
                       path_save_model + '\epoch_' + str(str_epoch) + '\model.pth')

            epochs_no_improve = 0
            min_val_loss = avg_val_loss

        else:

            epochs_no_improve += 1

            # Check if the epochs for which no improve have noticed at validation loss have exceeded patience value
            # in order to raise an early stopping
            if epochs_no_improve == patience:
                print_out = 'Early stopping raised at epoch: ' + str(epoch + 1) + '. Best model found in epoch: ' + str(
                    epoch + 1 - patience) + ' with validation loss: ' + str(epoch_val_losses[epoch - patience])
                print(print_out)

                with open(path_txt, 'a') as f:
                    f.write(print_out)

                early_stop = True

                # Load the model of epoch = current_epoch - patience , because this is the best model
                model.load_state_dict(
                    torch.load(path_save_model + '/epoch_' + str(epoch + 1 - patience) + '\model.pth'))

                # Evaluate this model in order to make the plots
                model.eval()
                _, preds_eval, yb_eval_all_true = evaluate(model, criterion, validation_generator, words_to_numbers)

        if (epoch + 1) % 1 == 0 or early_stop:

            if isinstance(criterion, type(nn.MultiLabelSoftMarginLoss())) or isinstance(criterion,
                                                                                        type(nn.CrossEntropyLoss())):
                y_eval_predicted = [(torch.max(torch.exp(batch_preds[0]), 1)[1]).data.cpu().numpy() for batch_preds in
                                    preds_eval]  # Returns the maximum values if the predictions
                # are of the form (Batches, Batch_size, Number_of_Classes)
                # This happens if there is not Softmax output layer

                # Create a list of tensors because the last batch might has different size of the rest and so they cannot be stacked
                preds_eval_tolistOfTensors = [preds_eval[batch_preds][0] for batch_preds in range(len(preds_eval) - 1)]

                # Stack all tensors except the last
                tensor_preds_eval = torch.stack(preds_eval_tolistOfTensors)

                # Create a list with the stacked tensors plus the last tensor
                tensor_preds_eval_plus_last_batch = [tensor_preds_eval, preds_eval[-1][0]]

                # Convert last Dense layer output to probabilities
                # Then, convert the resulting torches to a list of numpys
                # Next, reshape numpy in order not to take into account the number of batches
                list_probs = [np.reshape(((F.softmax(tens, dim=len(list(tens.size())) - 1)).numpy()), (-1, nm_classes))
                              for tens in tensor_preds_eval_plus_last_batch]

                # Finally stack the two resulting numpys
                probs = np.vstack(list_probs)

            y_eval_predicted = np.concatenate(y_eval_predicted).ravel()

            y_eval_all = [y_eval_batch.data.cpu().numpy() for y_eval_batch in yb_eval_all_true]
            y_eval_all = np.concatenate(y_eval_all).ravel()  # Flatten list of batches predictions

            roc_auc_scs, f1_macr = classific_report(y_eval_all, y_eval_predicted, probs, individual_words,
                                                    words_to_numbers, epoch + 1, epochs, early_stop, path_results,
                                                    path_txt)
            val_auc_rocs.append(roc_auc_scs)
            f1_macrs.append(f1_macr)

            # Print and save the results of the current epoch
            print_out = '\nEpoch: %d, avg_loss: %1.5f, avg_val_loss: %1.5f' % (epoch + 1, avg_train_loss, avg_val_loss)
            print(print_out)

            with open(path_txt, 'a') as f:
                f.write(print_out)

            if early_stop:
                break

    create_losses_plot(epoch + 1, avg_train_losses, epoch_val_losses, criterion_to_str,
                       path_results)  # Plot training and validation losses
    create_roc_auc_scores_plot(epoch + 1, val_auc_rocs, path_results)  # Plot ROC-AUC scores of validation set
    create_f1_macros_plot(epoch + 1, f1_macrs, path_results)  # Plot F1 macro scores of validation set

    # Remove the folder where the models with the min_val_loss are stored
    #shutil.rmtree(path_save_model)


    ################Make predictions for test dataset and print the results##############
    model.eval()
    _, preds_test, yb_test_all_true = evaluate(model, criterion, testing_generator, words_to_numbers)

    if isinstance(criterion, type(nn.MultiLabelSoftMarginLoss())) or isinstance(criterion, type(nn.CrossEntropyLoss())):
        y_test_predicted = [(torch.max(torch.exp(batch_preds[0]), 1)[1]).data.cpu().numpy() for batch_preds in
                            preds_test]  # Returns the maximum values if the predictions
        # are of the form (Batches, Batch_size, Number_of_Classes)
        # This happens if there is not Softmax output layer

        # Create a list of tensors because the last batch might has different size of the rest and so they cannot be stacked
        preds_test_tolistOfTensors = [preds_test[batch_preds][0] for batch_preds in range(len(preds_test) - 1)]

        # Stack all tensors except the last
        tensor_preds_test = torch.stack(preds_test_tolistOfTensors)

        # Create a list with the stacked tensors plus the last tensor
        tensor_preds_test_plus_last_batch = [tensor_preds_test, preds_test[-1][0]]

        # Convert last Dense layer output to probabilities
        # Then, convert the resulting torches to a list of numpys
        # Next, reshape numpy in order not to take into account the number of batches
        test_list_probs = [np.reshape(((F.softmax(tens, dim=len(list(tens.size())) - 1)).numpy()), (-1, nm_classes)) for
                           tens in tensor_preds_test_plus_last_batch]

        # Finally stack the two resulting numpys
        test_probs = np.vstack(test_list_probs)

    y_test_predicted = np.concatenate(y_test_predicted).ravel()

    y_test_all = [y_test_batch.data.cpu().numpy() for y_test_batch in yb_test_all_true]
    y_test_all = np.concatenate(y_test_all).ravel()  # Flatten list of batches predictions

    # Print and save the results for test dataset
    print_out = '\n\nResults for test dataset:'
    print(print_out)

    with open(path_txt, 'a') as f:
        f.write(print_out)

    roc_auc_scs, f1_macr = classific_report(y_test_all, y_test_predicted, test_probs, individual_words,
                                            words_to_numbers, epoch + 1, epochs, early_stop, path_results, path_txt,
                                            'test')

    print_out = '\nF1-macro score: %1.5f, ROC-AUC score: %1.5f' % (f1_macr, roc_auc_scs)