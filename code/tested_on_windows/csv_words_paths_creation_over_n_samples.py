import pandas as pd
import os
import csv
import io

type_of_dataset = 'test' #Choose a value between 'train', 'val' and 'test'
name = type_of_dataset+'_set.csv'

#Change the path according the location where csv with paths is
GSL_path = "C:/Users/georg/Documents/mine/εργασιες/μεταπτυχιακό/2ο εξάμηνο/Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα/1η εργασία\datasets/univ_patras/GSL_split/GSL_split/GSL_isolated/"
initial_data = pd.read_csv(
    GSL_path + type_of_dataset +"_greek_iso.csv",
    header=None)

paths = pd.DataFrame((initial_data[0].str.split("|")).str[0])
words = (initial_data[0].str.split("|")).str[1]

data = pd.concat([words, paths], axis=1)
data.columns = ['words', 'paths']

dict_words = {}
array_first_words_with_given_samples = []

if type_of_dataset == 'train':
    first_words = 10 #Change this value as the number of classes you want
    samples_lower_threshold = 500 #Change this number as the lower number of samples per words you want
    array_all_words = data['words'].unique()[:] #Get all the individual words

    temp_num_samples_per_word = 0

    count_words = 0
    for word in array_all_words:
        temp_samples_per_word = [data['paths'].loc[i] for i in range(data.shape[0]) if data['words'].loc[i] == word] #Get all the paths of the current word
        temp_num_samples_per_word = len(temp_samples_per_word) #Get the count of paths (samples) of the current words
        if temp_num_samples_per_word >= samples_lower_threshold:
            if temp_num_samples_per_word > samples_lower_threshold:
                dict_words[word] = temp_samples_per_word[:samples_lower_threshold].copy() #Keep only n samples of the current word where n=samples_lower_threshold to construct
                                                                                          #a balanced dataset
            else:
                dict_words[word] = temp_samples_per_word.copy()
            array_first_words_with_given_samples.append(word)
            count_words += 1
            if count_words >= first_words:
                break


else:
    #If type_of_dataset = 'val' or 'test', first it should have been created a train_set.csv and be located in GSL_path
    train_set = pd.read_csv(GSL_path + "train_set.csv")
    array_first_words_with_given_samples = train_set.columns.to_list()
    for word in array_first_words_with_given_samples:
        dict_words[word] = [data['paths'].loc[i] for i in range(data.shape[0]) if data['words'].loc[i] == word]



file = io.open(name, 'w', encoding='utf-8')
writer = csv.writer(file)
writer.writerow(array_first_words_with_given_samples)  # Write the words as names of the columns

column_list = []
for _, path in dict_words.items():
    column_list.append(path)

# Find the list with the max length. This means to find the word with the max paths
max_len = 0
for i in column_list:

    if len(i) > max_len:
        max_len = len(i)


for j in range(max_len):
    row_list = []
    for i in column_list:
        if len(i) > j:
            row_list.append(i[j])
        else:
            row_list.append('')

    writer.writerow(row_list)

file.close()
