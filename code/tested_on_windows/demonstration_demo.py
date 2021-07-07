import pandas as pd
import torch
import numpy as np
import os
import io
import time

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import cv2

from PIL import ImageFont, ImageDraw, Image


# Model Architecture
class Network(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layer_size, num_layers=1, num_filters=64, kernel_size=5, dilation=1, stride=1,
                 padding=0):
        super().__init__()

        self.num_classes = num_classes

        #### 1D CNN ###
        self.cnn_flag = True

        if self.cnn_flag:

            #1st 1DConv layer
            conv1_num_filters = 64
            conv1_kernel_size = kernel_size
            conv1_dilation = dilation
            conv1_padding = padding
            conv1_stride = stride
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv1_num_filters, kernel_size=conv1_kernel_size, dilation=conv1_dilation, padding=conv1_padding,
                                   stride=conv1_stride)

            #1st 1DMaxPooling layer
            maxpool1_kernel_size = kernel_size
            maxpool1_dilation = dilation
            maxpool1_padding = padding
            maxpool1_stride = maxpool1_kernel_size #Default for stride of maxPool1D is stride=kernel_size
            self.maxPool1 = nn.MaxPool1d(kernel_size=maxpool1_kernel_size, dilation=maxpool1_dilation, padding=maxpool1_padding, stride=maxpool1_stride)

            #2nd 1DConv layer
            #conv2_num_filters = 64
            #conv2_kernel_size = kernel_size
            #conv2_dilation = dilation
            #conv2_padding = padding
            #conv2_stride = stride
            #self.conv2 = nn.Conv1d(in_channels=conv1_num_filters, out_channels=conv2_num_filters, kernel_size=conv2_kernel_size,
            #                        dilation=conv2_dilation, padding=conv2_padding,
            #                        stride=conv1_stride)

            #2nd 1DMaxPooling layer
            #maxpool2_kernel_size = kernel_size
            #maxpool2_dilation = dilation
            #maxpool2_padding = padding
            #maxpool2_stride = maxpool2_kernel_size  # Default for stride of maxPool1D is stride=kernel_size
            #self.maxPool2 = nn.MaxPool1d(kernel_size=maxpool2_kernel_size, dilation=maxpool2_dilation,
            #                              padding=maxpool2_padding, stride=maxpool2_stride)


            #Calculate cnn output size
            conv1_out_l_size = int(((input_size + (2*conv1_padding) - (conv1_dilation*(conv1_kernel_size-1))-1)/conv1_stride)+1) #Where l is L of pytorch documentation for conv1D
            maxpool1_out_l_size = int(((conv1_out_l_size + (2*maxpool1_padding) - (maxpool1_dilation*(maxpool1_kernel_size-1))-1)/maxpool1_stride)+1)

            #conv2_out_l_size = int(((maxpool1_out_l_size + (2 * conv2_padding) - (
            #             conv2_dilation * (conv2_kernel_size - 1)) - 1) / conv2_stride) + 1)
            #maxpool2_out_l_size = int(((conv2_out_l_size + (2*maxpool2_padding) - (maxpool2_dilation*(maxpool2_kernel_size-1))-1)/maxpool2_stride)+1)


            input_size_for_next_layer = maxpool1_out_l_size*conv1_num_filters #maxpool2_out_l_size*conv2_num_filters



        ###LSTM OR GRU####
        self.num_layers = num_layers

        self.hidden_layer_size = hidden_layer_size

        if self.cnn_flag:
            inp_sz = input_size_for_next_layer
            #self.dropout_cnn_1 = nn.Dropout(0.2)
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

        #First Dense
        self.fc_1 = nn.Linear(hidden_layer_size, 128)

        # First Dropout
        self.dropout_1 = nn.Dropout(0.2)

        #Second Dense
        #self.fc_2 = nn.Linear(128, 64)

        #Second Dropout
        #self.dropout_2 = nn.Dropout(0.2)

        #Final Dense
        self.fc_output = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

        # self.softmax = nn.Softmax()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):

        if self.cnn_flag:
            #####CNN 1D######
            size_input = list(input_seq.size())

            cnn_input_seq = input_seq.double()
            cnn_input_seq = cnn_input_seq.view(size_input[0]*size_input[1], size_input[2]) #Flat first dim (batches) with second dim (seq)
            cnn_input_seq = cnn_input_seq.unsqueeze(1) #Add a dummy dimension as channel

            #1st cnn layer
            conv1_output = self.conv1(cnn_input_seq)
            maxpool1_output = self.maxPool1(conv1_output)
            out_relu_maxpool1 = self.relu(maxpool1_output)

            #First dropout for cnn layers
            #cnn_dropout1 = self.dropout_cnn_1(out_relu_maxpool1)

            #2nd cnn layer
            #conv2_output = self.conv2(cnn_dropout1)
            #maxpool2_output = self.maxPool1(conv2_output)
            #out_relu_maxpool2 = self.relu(maxpool2_output)

            #Final cnn output
            cnn_output_seq = out_relu_maxpool1.squeeze(1) #out_relu_maxpool2.squeeze(1)

            size_cnn_output_seq = list(cnn_output_seq.size())
            first_reshape_cnn_output_seq = cnn_output_seq.view(size_cnn_output_seq[0], size_cnn_output_seq[1]*size_cnn_output_seq[2])
            second_reshape_cnn_output_seq = first_reshape_cnn_output_seq.view(size_input[0], size_input[1], size_cnn_output_seq[1]*size_cnn_output_seq[2])

            #Dropout of final cnn layer
            final_dropout_cnn = self.final_dropout_cnn(second_reshape_cnn_output_seq)
            cnn_output_for_next_layer_input = final_dropout_cnn #cnn_output_for_next_layer_input = second_reshape_cnn_output_seq

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
            #final_lstm_drop = self.final_lstm_dropout(out_relu1)

            #Final LSTM output
            final_output = out_relu1 #final_lstm_drop


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

            #Final GRU dropout
            final_gru_drop = self.final_gru_dropout(out_relu1)

            # Final GRU output
            final_output = final_gru_drop #out_relu1

        ##### Dense layers #######

        #First Dense
        out_dense1 = self.fc_1(final_output)
        out_relu2 = self.relu(out_dense1)

        #First Dropout
        out_dropout1 = self.dropout_1(out_relu2)

        #Second Dense
        #out_dense2 = self.fc_2(out_relu2)
        #out_relu3 = self.relu(out_dense2)

        #Second Dropout
        #out_dropout2 = self.dropout_2(out_relu3)

        #Final Dense
        out_dense_output = self.fc_output(out_dropout1)

        # out_sigmoid = self.sigmoid(out_dense2)
        # out_softmax = self.softmax(out_dense2) #It isnt needed for MultiLabelSoftMarginLoss() nor for CrossEntropy

        return out_dense_output #out_dense2  # out_sigmoid  # out_softmax



if __name__ == '__main__':

    #Choose an example as integer from 0-11. Each one correspond to a different word.
    example = 0

    list_of_dir_frames = ['health1_signer3_rep3_glosses/glosses0000',
                          'health2_signer3_rep3_glosses/glosses0004',
                          'kep1_signer3_rep3_glosses/glosses0065',
                          'kep2_signer3_rep5_glosses/glosses0050',
                          'health1_signer3_rep3_glosses/glosses0048',
                          'health1_signer3_rep3_glosses/glosses0012',
                          'health2_signer3_rep3_glosses/glosses0074',
                          'health1_signer3_rep3_glosses/glosses0049',
                          'health1_signer3_rep3_glosses/glosses0033',
                          'health1_signer3_rep3_glosses/glosses0030',
                          'health1_signer3_rep3_glosses/glosses0006',
                          'health3_signer3_rep3_glosses/glosses0012'
                          ]
    actual_words_of_each_dir_frames = ['ΓΕΙΑ', 'ΕΓΩ(1)', 'ΓΙΑ', 'ΕΣΥ', 'ΕΝΤΑΞΕΙ', 'ΣΥΝ', 'ΕΥΧΑΡΙΣΤΩ', 'ΚΑΛΟ', 'ΒΙΒΛΙΟ', 'ΕΧΩ', 'ΧΡΕΙΑΖΟΜΑΙ', 'ΧΑΡΤΙ']
    parent_root_dir_frames = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\datasets\univ_patras\GSL_split\GSL_split\GSL_isolated\Greek_isolated\GSL_isol'
    dir_frames = parent_root_dir_frames + '/' + list_of_dir_frames[example]
    dir_model = r'C:\Users\georg\Documents\mine\18BC~1\AB01~1\28B48~1\C4C6~1\105AF~1\results\12_FIR~1\FRAMES~1\SAMPLI~1\CROSSE~1\points\1_1DCN~2\2NDEXP~1\TEMP_S~1\epoch_42'
    dir_font = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\ubuntu-font-family-0.80'

    filter_flag = False  # Change this value if you dont want to apply image preprocessing
    face_points_flag = True  # Change this value to True if you want face points or to False if do not want them
    face_point_sample_rate = 5  # Change this value to determine the sampling rate of face points
    max_num_frames = 10  # Maximum number of frames to keep for each gesture
    hidden_size = 100 #Number of units of GRU/LSTM layers
    nm_layers = 1
    nm_classes = 12 #Number of individual words
    individual_classes = ['ΓΕΙΑ', 'ΕΓΩ(1)', 'ΓΙΑ', 'ΕΣΥ', 'ΕΝΤΑΞΕΙ', 'ΣΥΝ', 'ΕΥΧΑΡΙΣΤΩ', 'ΚΑΛΟ', 'ΒΙΒΛΙΟ', 'ΕΧΩ', 'ΧΡΕΙΑΖΟΜΑΙ', 'ΧΑΡΤΙ']


    # Get all the files of the current path
    frame_list = os.listdir(dir_frames)

    # Exclude every file which is not a frame (which has not .jpg extension)
    frame_list = [frame for frame in frame_list if (frame.split('.'))[-1] == 'jpg']

    frames_paths = [dir_frames + '/' + frame for frame in frame_list]

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    #Start to read the frames, extract the points and make predictions with the pretrained model
    with mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        # Create a list for the columns' names of csv
        point_list = []

        count_hand_points = 0
        for idx, landmark in enumerate(mp_holistic.HandLandmark):
            point_list.append("Right_x_" + landmark.name)
            point_list.append("Right_y_" + landmark.name)
            count_hand_points += 1

        for idx, landmark in enumerate(mp_holistic.HandLandmark):
            point_list.append("Left_x_" + landmark.name)
            point_list.append("Left_y_" + landmark.name)

        if face_points_flag:
            count_face_points = 0
            for idx, landmark in enumerate(mp_holistic.FACE_CONNECTIONS):
                if idx % face_point_sample_rate == 0:
                    point_list.append(str(idx) + "_x")
                    point_list.append(str(idx) + "_y")
                    count_face_points += 1

        count_pose_points = 0
        for idx, landmark in enumerate(mp_holistic.PoseLandmark):
            if landmark.name == 'LEFT_SHOULDER' or landmark.name == 'RIGHT_SHOULDER':
                point_list.append("x_" + landmark.name)
                point_list.append("y_" + landmark.name)
                count_pose_points += 1

        # Create the dataframe and write the column names
        dataframe_points = pd.DataFrame(columns=point_list)

        # Instantiate Network
        model = Network(len(point_list), nm_classes, hidden_size, nm_layers)
        if model.cnn_flag:
            model = model.double()

        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True

        #Load pretrained model
        model.load_state_dict(torch.load(dir_model + '\model.pth'))

        for id_frame, frame_path in enumerate(frames_paths):

            frame = cv2.imdecode(np.fromfile(u'' + frame_path, np.uint8), cv2.IMREAD_UNCHANGED)

            if filter_flag:
                scale_percent = 400  # percent of original size
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)
                dim = (width, height)

                blur_image = cv2.blur(frame, (3, 3))

                # resize image and apply interpolation
                resized_image = cv2.resize(blur_image, dim, interpolation=cv2.INTER_AREA)

                image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

            else:

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            imageHeight, imageWidth, _ = image.shape

            image.flags.writeable = False
            results = holistic.process(image)

            list_to_dataframe_row = []

            # Right hand info: 21 points
            right_hand_list = []
            if results.right_hand_landmarks:
                # print("Right_hand:")
                for idx, landmark in enumerate(mp_holistic.HandLandmark):
                    # print(landmark.name)
                    # print(results.right_hand_landmarks.landmark[landmark])
                    res = results.right_hand_landmarks.landmark[landmark]
                    right_hand_list.extend([res.x, res.y])
            else:
                right_hand_list = [0.0] * (count_hand_points * 2)

            list_to_dataframe_row.extend(right_hand_list)

            # Left hand info: 21 points
            left_hand_list = []
            if results.left_hand_landmarks:
                # print("Left hand:")
                for idx, landmark in enumerate(mp_holistic.HandLandmark):
                    # print(landmark.name)
                    # print(results.left_hand_landmarks.landmark[landmark])
                    res = results.left_hand_landmarks.landmark[landmark]
                    left_hand_list.extend([res.x, res.y])
            else:
                left_hand_list = [0.0] * (count_hand_points * 2)

            list_to_dataframe_row.extend(left_hand_list)

            if face_points_flag:
                # Face info: 124 points
                face_list = []
                if results.face_landmarks:
                    # print("Face:")
                    for idx, landmark in enumerate(mp_holistic.FACE_CONNECTIONS):
                        if idx % face_point_sample_rate == 0:
                            # print(idx)
                            # print(landmark)
                            # print(results.face_landmarks.landmark[idx])
                            res = results.face_landmarks.landmark[idx]
                            face_list.extend([res.x, res.y])
                else:
                    face_list = [0.0] * (count_face_points * 2)

                list_to_dataframe_row.extend(face_list)

            # Pose info: 33 points
            pose_list = []
            if results.pose_landmarks:
                # print("Pose:")
                for idx, landmark in enumerate(mp_holistic.PoseLandmark):
                    if landmark.name == 'LEFT_SHOULDER' or landmark.name == 'RIGHT_SHOULDER':
                        # print(idx)
                        # print(landmark.name)
                        # print(results.pose_landmarks.landmark[landmark])
                        res = results.pose_landmarks.landmark[landmark]
                        pose_list.extend([res.x, res.y])
            else:
                pose_list = [0.0] * (count_pose_points * 2)

            list_to_dataframe_row.extend(pose_list)

            # Add a row to dataframe with data of the current frame
            current_frame_points_dataframe = pd.DataFrame([list_to_dataframe_row], columns=point_list)

            dataframe_points = dataframe_points.append(current_frame_points_dataframe).reset_index().drop('index', axis=1)

            # Uncomment the below if you want to annotate the initial image

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            annotated_image = image.copy()

            mp_drawing.draw_landmarks(
                annotated_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)

            mp_drawing.draw_landmarks(
                annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            mp_drawing.draw_landmarks(
                annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)



            # Uncomment if you want to write annotated images
            """
            if not os.path.exists(file_path + '/' + training_folder + "/" + csv_name):
                os.makedirs(file_path + '/' + training_folder + "/" + csv_name)

            # Write image in this way because imwrite does not support greek letters in path
            is_success, im_buff_arr = cv2.imencode('.jpg', annotated_image)
            im_buff_arr.tofile(file_path + '/' + training_folder + '/' + csv_name + '/' + str(id_frame) + '.png')
            """

            # Uncomment if you want to see images
            winname = 'MediaPipe Annotation'
            cv2.namedWindow(winname)  # Create a named window
            cv2.moveWindow(winname, 800, 10)  # Move it to (40,30)
            cv2.imshow(winname, annotated_image)

            if id_frame < max_num_frames - 1:
                winname_ = 'Original image'
                cv2.namedWindow(winname_)  # Create a named window
                cv2.moveWindow(winname_, 10, 10)  # Move it to the desired position
                cv2.imshow('Original image', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

            time.sleep(0.5)

            if id_frame >= max_num_frames-1:
                model.eval()


                #Transform the last 'max_num_frames' frames to Dataframe then to numpy, then to torch tensor and make a prediction
                landmarks = dataframe_points.iloc[id_frame - (max_num_frames-1):, :]
                landmarks = np.array([landmarks])
                xb = torch.from_numpy(landmarks)

                with torch.no_grad():
                    predictions = model(xb.to(device))

                probabilities = F.softmax(predictions, dim=len(predictions)).numpy()
                predicted_class_to_number = np.argmax(probabilities)
                predicted_class_to_word = individual_classes[predicted_class_to_number]

                #Show predicted class to the original image

                font = ImageFont.truetype(dir_font+'/Ubuntu-R.ttf', 32)
                img_pil = Image.fromarray(image)

                #Predicted class text
                draw = ImageDraw.Draw(img_pil)
                b, g, r, a = 255, 255, 255, 0
                draw.text((70, 5), 'Predicted: '+predicted_class_to_word, font=font, fill=(b, g, r, a))

                #Actual class text
                draw = ImageDraw.Draw(img_pil)
                b, g, r, a = 255, 255, 255, 0
                draw.text((350, 5), 'Actual: ' + actual_words_of_each_dir_frames[example], font=font, fill=(b, g, r, a))

                image = np.array(img_pil)

                winname_ = 'Original image'
                cv2.namedWindow(winname_)  # Create a named window
                cv2.moveWindow(winname_, 10, 10)  # Move it to (40,30)
                cv2.imshow('Original image', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break



