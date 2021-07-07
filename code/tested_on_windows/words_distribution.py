import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import io

type_of_dataset = 'train' #Choose a value between 'train', 'valid' and 'test'
name = type_of_dataset+'_set.csv'
mode = 'from_modified_data' #'from_initial_data'

#Change the path according the location where csv with paths is
GSL_path = "C:/Users/georg/Documents/mine/εργασιες/μεταπτυχιακό/2ο εξάμηνο/Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα/1η εργασία\datasets/univ_patras/GSL_split/GSL_split/GSL_isolated/"
initial_data = pd.read_csv(
    GSL_path + type_of_dataset +"_greek_iso.csv",
    header=None)

paths = pd.DataFrame((initial_data[0].str.split("|")).str[0])
words = (initial_data[0].str.split("|")).str[1]

data = pd.concat([words, paths], axis=1)
data.columns = ['words', 'paths']
if type_of_dataset == 'train':

    if mode == 'from_initial_data':
        first_words = len(data['words'].unique()[:]) #10 #Change this value as the number of classes you want
        array_first_words = data['words'].unique()[:first_words]

    elif mode == 'from_modified_data':
        train_set = pd.read_csv(GSL_path + "train_set.csv")
        array_first_words = train_set.columns.to_list()
        first_words = len(array_first_words)

else:
    #If type_of_dataset = 'val' or 'test', first it should have been created a train_set.csv and be located in GSL_path
    train_set = pd.read_csv(GSL_path + "train_set.csv")
    array_first_words = train_set.columns.to_list()


dict_words = {}

for word in array_first_words:
    dict_words[word] = [data['paths'].loc[i] for i in range(data.shape[0]) if data['words'].loc[i] == word]


column_list = []
for _, path in dict_words.items():
    column_list.append(len(path))


fig, ax1 = plt.subplots(1, 1, figsize=(25, 10))

x_axis_list = np.arange(0, len(column_list))

ax1.bar(list(dict_words.keys()), column_list)
ax1.set_title('Number of samples for the ' + str(first_words) + ' first words')
ax1.set_ylabel('Counts', fontsize=12)
ax1.set_xlabel('Words', fontsize=12)

for index, data in enumerate(column_list):
    plt.text(x=index, y=data+1, s=f"{data}", fontdict=dict(fontsize=20))
plt.tight_layout()

plt.show()