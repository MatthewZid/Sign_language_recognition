import pandas as pd
import os
import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import csv
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

filter_flag = False #Change this value if you dont want to apply image preprocessing
face_points_flag = True #Change this value if you dont want to create face points
face_sample_rate = 5 #If 'face_points_flag' is True then you have to specify the sample rate of face points. If you want all the face point set it equal to 1
max_num_frames = 10 #Maximum number of frames to keep for each gesture
padding = 'zero' #Change this value corresponding to the current technique

type_of_dataset = 'test' #Choose a value between 'train', 'val' and 'test'

#Change the path where your csv file is
file_path = 'C:/Users/georg/Documents/mine/εργασιες/μεταπτυχιακό/2ο εξάμηνο/Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα/1η εργασία/datasets/univ_patras/GSL_split/GSL_split/GSL_isolated'

#Get the train_set.csv
words_paths_data = pd.read_csv(file_path + '/' + type_of_dataset + '_set.csv')


GSL_iso_path = file_path + '/' + 'Greek_isolated/GSL_isol/' #Change this path where the directory of frames paths are


#Create a csv file with points values for each set of frames in each path
count_completed = 0
overall_samples = np.sum(words_paths_data.count())
data_shape = words_paths_data.shape

for row, column in [(row, column) for row in range(data_shape[0]) for column in range(data_shape[1])]:
    temp_curr_path = (words_paths_data.loc[row])[column]
    if isinstance(temp_curr_path, float): #If there is no other path (which means all the others are NaN values) for this column, then skip it
        continue
    else:
        curr_path = temp_curr_path

    #Get all the files of the current path
    frame_list = os.listdir(GSL_iso_path+curr_path)

    #Exclude every file which is not a frame (which has not .jpg extension)
    frame_list = [frame for frame in frame_list if (frame.split('.'))[-1] == 'jpg']

    frames_paths = [GSL_iso_path+curr_path+'/'+frame for frame in frame_list]
    if len(frames_paths) > max_num_frames: #Check if the number of frames for the current gesture is greater than maximum number of frames that has been chosen

        exclude_last = len(frames_paths) - 10 #Calculate the number of frames to be excluded
        frames_paths = frames_paths[:-exclude_last] #Exclude extra frames

    with mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        # Create a list for the columns' names of csv
        point_list = []

        count_hand_points = 0
        for idx, landmark in enumerate(mp_holistic.HandLandmark):
            point_list.append("Right_angle_" + landmark.name)
            point_list.append("Right_magn_" + landmark.name)
            count_hand_points += 1

        for idx, landmark in enumerate(mp_holistic.HandLandmark):
            point_list.append("Left_angle_" + landmark.name)
            point_list.append("Left_magn_" + landmark.name)


        if face_points_flag:
            count_face_points = 0
            for idx, landmark in enumerate(mp_holistic.FACE_CONNECTIONS):
                if idx % face_sample_rate == 0:
                    point_list.append(str(idx) + "_angle")
                    point_list.append(str(idx) + "_magn")
                    count_face_points += 1


        count_pose_points = 0
        for idx, landmark in enumerate(mp_holistic.PoseLandmark):
            if landmark.name == 'LEFT_SHOULDER' or landmark.name == 'RIGHT_SHOULDER':
                point_list.append("angle_"+landmark.name)
                point_list.append("magn_"+landmark.name)
                count_pose_points += 1


        # Create the csv and write the column names
        dataset_folder = type_of_dataset + '_points_data_num_frames_' + str(max_num_frames-1) + '_'+padding
        if not os.path.exists(file_path + '/' + dataset_folder):
            os.makedirs(file_path + '/' + dataset_folder)

        csv_name = curr_path.replace('/', '_')
        with open(file_path + '/' + dataset_folder + "/" + csv_name + ".csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(point_list)


        previous_right_hand_list = []
        previous_left_hand_list = []
        if face_points_flag:
            previous_face_list = []
        previous_pose_list = []

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

            list_to_csv_row = []

            # Right hand info: 21 points
            right_hand_list = []
            right_hand_angle_magn_list = []
            if results.right_hand_landmarks:
                # print("Right_hand:")
                for idx, landmark in enumerate(mp_holistic.HandLandmark):
                    # print(landmark.name)
                    # print(results.right_hand_landmarks.landmark[landmark])
                    res = results.right_hand_landmarks.landmark[landmark]
                    if id_frame > 0:

                        #Calculate the angle between the current hand point in this frame and the previous frame
                        if (res.x - previous_right_hand_list[2*idx]) != 0:
                            angle = math.atan((res.y - previous_right_hand_list[(2*idx)+1]) / (res.x - previous_right_hand_list[2*idx]))
                        else:
                            angle = 0
                        right_hand_angle_magn_list.append(angle)

                        # Calculate the magnitude of the current hand point for this frame and the previous frame
                        magn = math.sqrt(((res.x - previous_right_hand_list[2*idx])**2) + ((res.y - previous_right_hand_list[(2*idx)+1])**2))
                        right_hand_angle_magn_list.append(magn)

                    right_hand_list.extend([res.x, res.y])
            else:
                right_hand_list = [0] * (count_hand_points * 2)
                if id_frame > 0:

                    for idx in range(count_hand_points):

                        # Calculate the angle between the current hand point in this frame and the previous frame
                        if previous_right_hand_list[2 * idx] != 0:
                            angle = math.atan((0 - previous_right_hand_list[(2 * idx) + 1]) / (0 - previous_right_hand_list[2 * idx]))
                        else:
                            angle = 0
                        right_hand_angle_magn_list.append(angle)

                        # Calculate the magnitude of the current hand point for this frame and the previous frame
                        magn = math.sqrt(((0 - previous_right_hand_list[2 * idx]) ** 2) + ((0 - previous_right_hand_list[(2 * idx) + 1]) ** 2))
                        right_hand_angle_magn_list.append(magn)

            if id_frame > 0:
                list_to_csv_row.extend(right_hand_angle_magn_list)

            #Update the list of the previous right hand points
            previous_right_hand_list = right_hand_list.copy()


            # Left hand info: 21 points
            left_hand_list = []
            left_hand_angle_magn_list = []
            if results.left_hand_landmarks:
                # print("Left hand:")
                for idx, landmark in enumerate(mp_holistic.HandLandmark):
                    # print(landmark.name)
                    # print(results.left_hand_landmarks.landmark[landmark])
                    res = results.left_hand_landmarks.landmark[landmark]
                    if id_frame > 0:

                        #Calculate the angle between the current hand point in this frame and the previous frame
                        if (res.x - previous_left_hand_list[2*idx]) != 0:
                            angle = math.atan((res.y - previous_left_hand_list[(2*idx)+1]) / (res.x - previous_left_hand_list[2*idx]))
                        else:
                            angle = 0
                        left_hand_angle_magn_list.append(angle)

                        # Calculate the magnitude of the current hand point for this frame and the previous frame
                        magn = math.sqrt(((res.x - previous_left_hand_list[2*idx])**2) + ((res.y - previous_left_hand_list[(2*idx)+1])**2))
                        left_hand_angle_magn_list.append(magn)

                    left_hand_list.extend([res.x, res.y])
            else:
                left_hand_list = [0] * (count_hand_points * 2)
                if id_frame > 0:

                    for idx in range(count_hand_points):

                        # Calculate the angle between the current hand point in this frame and the previous frame
                        if previous_left_hand_list[2 * idx] != 0:
                            angle = math.atan((0 - previous_left_hand_list[(2 * idx) + 1]) / (0 - previous_left_hand_list[2 * idx]))
                        else:
                            angle = 0
                        left_hand_angle_magn_list.append(angle)

                        # Calculate the magnitude of the current hand point for this frame and the previous frame
                        magn = math.sqrt(((0 - previous_left_hand_list[2 * idx]) ** 2) + ((0 - previous_left_hand_list[(2 * idx) + 1]) ** 2))
                        left_hand_angle_magn_list.append(magn)

            if id_frame > 0:
                list_to_csv_row.extend(left_hand_angle_magn_list)

            # Update the list of the previous left hand points
            previous_left_hand_list = left_hand_list.copy()



            if face_points_flag:
                # Face info: 124 points
                face_list = []
                face_angle_magn_list = []
                if results.face_landmarks:
                    # print("Face:")
                    for idx, landmark in enumerate(mp_holistic.FACE_CONNECTIONS):
                        if idx % face_sample_rate == 0:
                            # print(idx)
                            # print(landmark)
                            # print(results.face_landmarks.landmark[idx])
                            res = results.face_landmarks.landmark[idx]
                            if id_frame > 0:
                                # Calculate the angle between the current face point in this frame and the previous frame
                                if (res.x - previous_face_list[2 * int(idx/face_sample_rate)]) != 0:
                                    angle = math.atan((res.y - previous_face_list[(2 * int(idx/face_sample_rate)) + 1]) / (res.x - previous_face_list[2 * int(idx/face_sample_rate)]))
                                else:
                                    angle = 0
                                face_angle_magn_list.append(angle)

                                # Calculate the magnitude of the current face point for this frame and the previous frame
                                magn = math.sqrt(((res.x - previous_face_list[2 * int(idx/face_sample_rate)]) ** 2) + ((res.y - previous_face_list[(2 * int(idx/face_sample_rate)) + 1]) ** 2))
                                face_angle_magn_list.append(magn)

                            face_list.extend([res.x, res.y])
                else:
                    face_list = [0] * (count_face_points * 2)
                    if id_frame > 0:

                        for idx in range(count_face_points):

                            # Calculate the angle between the current face point in this frame and the previous frame
                            if previous_face_list[2 * idx] != 0:
                                angle = math.atan((0 - previous_face_list[(2 * idx) + 1]) / (0 - previous_face_list[2 * idx]))
                            else:
                                angle = 0
                            face_angle_magn_list.append(angle)

                            # Calculate the magnitude of the current face point for this frame and the previous frame
                            magn = math.sqrt(((0 - previous_face_list[2 * idx]) ** 2) + ((0 - previous_face_list[(2 * idx) + 1]) ** 2))
                            face_angle_magn_list.append(magn)

                if id_frame > 0:
                    list_to_csv_row.extend(face_angle_magn_list)

                # Update the list of the previous face points
                previous_face_list = face_list.copy()


            #Pose info: 33 points
            pose_list = []
            pose_angle_magn_list = []
            if results.pose_landmarks:
                #print("Pose:")
                pose_points_counter = 0
                for idx, landmark in enumerate(mp_holistic.PoseLandmark):
                    if landmark.name == 'LEFT_SHOULDER' or landmark.name == 'RIGHT_SHOULDER':
                        #print(idx)
                        #print(landmark.name)
                        #print(results.pose_landmarks.landmark[landmark])
                        res = results.pose_landmarks.landmark[landmark]
                        if id_frame > 0:

                            # Calculate the angle between the current pose point in this frame and the previous frame
                            if (res.x - previous_pose_list[2 * pose_points_counter]) != 0:
                                angle = math.atan((res.y - previous_pose_list[(2 * pose_points_counter) + 1]) / (res.x - previous_pose_list[2 * pose_points_counter]))
                            else:
                                angle = 0
                            pose_angle_magn_list.append(angle)

                            # Calculate the magnitude of the current pose point for this frame and the previous frame
                            magn = math.sqrt(((res.x - previous_pose_list[2 * pose_points_counter]) ** 2) + ((res.y - previous_pose_list[(2 * pose_points_counter) + 1]) ** 2))
                            pose_angle_magn_list.append(magn)

                        pose_list.extend([res.x, res.y])
                        pose_points_counter += 1
            else:
                pose_list = [0] * (count_pose_points*2)
                if id_frame > 0:

                    for idx in range(count_pose_points):

                        # Calculate the angle between the current pose point in this frame and the previous frame
                        if previous_pose_list[2 * idx] != 0:
                            angle = math.atan((0 - previous_pose_list[(2 * idx) + 1]) / (0 - previous_pose_list[2 * idx]))
                        else:
                            angle = 0
                        pose_magn_list.append(angle)

                        # Calculate the magnitude of the current pose point for this frame and the previous frame
                        magn = math.sqrt(((0 - previous_pose_list[2 * idx]) ** 2) + ((0 - previous_pose_list[(2 * idx) + 1]) ** 2))
                        pose_angle_magn_list.append(magn)

            if id_frame > 0:
                list_to_csv_row.extend(pose_list)

            # Update the list of the previous pose points
            previous_pose_list = pose_list.copy()

            if id_frame > 0:
                # Add a row to csv with data of the current frame
                with open(file_path + '/' + dataset_folder + "/" + csv_name + ".csv", 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(list_to_csv_row)


            #Uncomment the below if you want to annotate the initial image
            """
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

            """


            #Uncomment if you want to write annotated images
            """
            if not os.path.exists(file_path + '/' + training_folder + "/" + csv_name):
                os.makedirs(file_path + '/' + training_folder + "/" + csv_name)
            
            # Write image in this way because imwrite does not support greek letters in path
            is_success, im_buff_arr = cv2.imencode('.jpg', annotated_image)
            im_buff_arr.tofile(file_path + '/' + training_folder + '/' + csv_name + '/' + str(id_frame) + '.png')
            """


            #Uncomment if you want to see images
            """cv2.imshow('MediaPipe Hands', annotated_image)
            if cv2.waitKey(5) & 0xFF == 27:
                break"""

        count_completed += 1
        remains = overall_samples - count_completed
        print(csv_name + '.csv was creates successfully! Totally completed: ' + str(count_completed) + ". Remain: " + str(remains))

