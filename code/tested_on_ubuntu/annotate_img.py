import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import sys
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

typename = str(sys.argv[1])

undersampled = True
final_dataset_file = ''
imgpath = ''

ANNOTWIDTH = 648
ANNOTHEIGHT = 480
CHANNELS = 3

if undersampled and typename == 'train':
    final_dataset_file = '../' + typename + '_dataset_undersampled.csv'
    imgpath = '../annotated_imgs/' + typename + '_undersampled'
elif undersampled and typename == 'val':
    print('Cannot write undersampled and validation!')
    sys.exit(1)
else:
    final_dataset_file = '../' + typename + '_dataset.csv'
    imgpath = '../annotated_imgs/' + typename

root_video_path = r'../GSL_isol'

# keep connections accordingly
kept_pose_connections = []
keepindx = list(range(5))
keepindx.append(6)
keepindx.extend([11,12])
keepindx.extend([14,15,16])
keepindx.extend(list(range(19,23)))
keepindx.extend([25,26])
keepindx.append(28)
keepindx.extend([30,31,32])
keepindx.append(34)

ind = 0
for con in mp_holistic.POSE_CONNECTIONS:
    if ind in keepindx:
        kept_pose_connections.append(con)
    ind += 1
kept_pose_connections = frozenset(kept_pose_connections)

def extract_points(file_list, dirpath):
    frame = 0

    # For static images:
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2) as holistic:
        for _, file in enumerate(file_list):
            image = cv2.imread(file)
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
            temp = results.pose_landmarks.landmark[23:len(results.pose_landmarks.landmark)]
            for i in range(len(temp)):
                results.pose_landmarks.landmark.remove(temp[i])
            del temp
            
            # Draw pose, left and right hands on the image.
            # annotated_image = image.copy()
            annotated_image = np.zeros((ANNOTHEIGHT,ANNOTWIDTH,CHANNELS), dtype=np.uint8)

            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, kept_pose_connections)
            mp_drawing.draw_landmarks(
                annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # resize image
            annotated_image = cv2.resize(annotated_image, (128,128), interpolation = cv2.INTER_AREA)

            cv2.imwrite(os.path.join(dirpath, 'frame_{:04d}.png'.format(frame)), annotated_image)
            frame += 1
            break

    print("Processed dir: {}".format(dirpath))
    print("Total processed frames: {:d}\n".format(frame))

# prepare video paths
with open(final_dataset_file, "r") as datf:
    csvreader = csv.reader(datf, delimiter=",")
    for row in csvreader:
        video_path = row[0]
        total_path = os.path.join(root_video_path, video_path)
        file_list = [os.path.join(total_path, name) for name in os.listdir(total_path) if os.path.isfile(os.path.join(total_path, name))]
        file_list.sort()
        dirpath = os.path.join(imgpath, video_path)
        try:
            os.makedirs(dirpath)
        except OSError as error:
            print(error)
            sys.exit(1)
        
        extract_points(file_list, dirpath)
        break