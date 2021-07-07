import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import sys
import math

import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

typename = 'test'
max_num_frames = 10
final_dataset_file = ''
imgpath = ''

ANNOTWIDTH = 648
ANNOTHEIGHT = 480
CHANNELS = 3

final_dataset_file = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\datasets\univ_patras\GSL_split\GSL_split\GSL_isolated' + r'\\' + typename + '_set.csv'
imgpath = 'C:/Users/georg/Documents/mine/εργασιες/μεταπτυχιακό/2ο εξάμηνο/Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα/1η εργασία/annotated_imgs' + '_frames_' + str(max_num_frames) + '/' + typename

root_video_path = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\datasets\univ_patras\GSL_split\GSL_split\GSL_isolated\Greek_isolated\GSL_isol'

# keep connections accordingly
kept_pose_connections = []
keepindx = list(range(8)) #0-7
keepindx.append(9)
keepindx.append(11)
keepindx.extend([14, 16])
keepindx.append(19)
keepindx.extend([23, 24, 25, 26])
keepindx.extend([28, 29, 30, 33])


kept_pose_landmarks = []
ind = 0
for con in mp_holistic.POSE_CONNECTIONS:

    if ind in keepindx:
        kept_pose_connections.append(con)
        for elem in con:

            if elem.name not in kept_pose_landmarks:
                kept_pose_landmarks.append(elem.name)
    ind += 1
kept_pose_connections = frozenset(kept_pose_connections)



def extract_points(file_list, dirpath):
    frame = 0

    # For static images:
    with mp_holistic.Holistic(
            static_image_mode=True) as holistic:
        for _, file in enumerate(file_list):

            image = cv2.imdecode(np.fromfile(u'' + file, np.uint8), cv2.IMREAD_UNCHANGED)

            # image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # if results.pose_landmarks:
            #     print(
            #         f'Nose coordinates: ('
            #         f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
            #         f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
            #     )

            # remove unwanted landmarks
            temp = []

            for landmark in mp_holistic.PoseLandmark:
                if (landmark.name not in kept_pose_landmarks) and (landmark.name not in temp):
                    temp.append(results.pose_landmarks.landmark[landmark])


            for i in range(len(temp)):
                results.pose_landmarks.landmark.remove(temp[i])
            del temp

            # Draw pose, left and right hands on the image.
            # annotated_image = image.copy()
            annotated_image = np.zeros((ANNOTHEIGHT, ANNOTWIDTH, CHANNELS), dtype=np.uint8)

            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, kept_pose_connections)
            mp_drawing.draw_landmarks(
                annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # resize image
            #annotated_image = cv2.resize(annotated_image, (128, 128), interpolation=cv2.INTER_AREA)

            #cv2.imwrite((os.path.join(dirpath, 'frame_{:04d}.png'.format(frame))), annotated_image) #It does not work when utf8 characters are presented in the path. Instead of that, use the two lines below
            is_success, im_buf_arr = cv2.imencode(".jpg", annotated_image)
            im_buf_arr.tofile(os.path.join(dirpath, 'frame_{:04d}.jpg'.format(frame)))

            frame += 1
            if frame == max_num_frames:
                break


    print("Processed dir: {}".format(dirpath))
    print("Total processed frames: {:d}\n".format(frame))


# prepare video paths
csv_file = pd.read_csv(final_dataset_file)
for idx, col in enumerate(csv_file.columns):
    for index, row in csv_file.iterrows():

        try:

            if math.isnan(row[idx]):
                break

        except:

            video_path = row[idx]
            total_path = os.path.join(root_video_path, video_path)
            file_list = [os.path.join(total_path, name) for name in os.listdir(total_path) if
                         os.path.isfile(os.path.join(total_path, name))]
            file_list.sort()
            dirpath = os.path.join(imgpath, video_path)
            try:
                os.makedirs(dirpath)
            except OSError as error:
                print(error)
                sys.exit(1)
        
            extract_points(file_list, dirpath)
