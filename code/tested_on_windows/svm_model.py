import os
import numpy as np
import pandas as pd
import json
import io

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

import seaborn as sn
import matplotlib.pyplot as plt

import pickle

def dataset_creation(csv_words_paths, landmarks_csvs_dir):

    words_paths = pd.read_csv(csv_words_paths)
    landmarks_dir = landmarks_csvs_dir
    max_len_frame = int((landmarks_dir.split('\\'))[-1].split('_')[-2])
    individual_words = words_paths.columns
    individual_words = individual_words.to_list()
    words_to_numbers = np.arange(
        len(individual_words)).tolist()  # The representation of each word as a number is just its index in individual_words list
    count_samples = []
    total_count = 0
    for col in words_paths.columns:
        count_samples.append(words_paths[col].count())
        total_count += words_paths[col].count()


    for idx in range(total_count):

        # Each column does not have equal samples (there are NaN values)
        # So, let calculate the cumulative length column by column in order to return the sample with
        # the cumulative 'idx'
        cumulative_column_length = 0
        for column in words_paths.columns:
            column_idx = 0
            column_length = words_paths[column].count()
            cumulative_column_length += column_length
            if idx < cumulative_column_length:
                column_idx = idx - (cumulative_column_length - column_length)
                break

        word_path_name = words_paths[column].iloc[column_idx]
        word_path_name = word_path_name.replace('/', '_')

        word_path = os.path.join(landmarks_dir, word_path_name)

        landmarks = pd.read_csv(word_path + '.csv')
        landmarks = np.array([landmarks])

        initial_row_shape = landmarks.shape[1]  # If the the shape[1] is 0 then the array should be converted to np.float
        initial_column_shape = landmarks.shape[2]
        cur_row_shape = landmarks.shape[1]
        if abs(max_len_frame - landmarks.shape[1]) != 0:

            landmarks = np.append((landmarks[0]),
                                  [[0] * landmarks.shape[2] for _ in range(max_len_frame - landmarks.shape[1])],
                                  axis=0)
            landmarks = np.array([landmarks])
            cur_row_shape = landmarks.shape[1]

            if initial_row_shape == 0:
                landmarks = np.vstack(landmarks[:, :, :]).astype(np.float)
                cur_row_shape = landmarks.shape[0]


        landmarks = np.reshape(landmarks, (1, cur_row_shape*initial_column_shape))


        word_label = column
        word_index = individual_words.index(word_label)
        word_number_label = words_to_numbers[word_index]

        if idx == 0:
            X_array = landmarks
            y_array = np.array([word_label])
            y_array_numbers = np.array([word_number_label])
        else:
            X_array = np.append(X_array, landmarks, axis=0)
            y_array = np.append(y_array, np.array([word_label]), axis=0)
            y_array_numbers = np.append(y_array_numbers, np.array([word_number_label]), axis=0)


    return X_array, y_array, y_array_numbers, individual_words


if __name__ == '__main__':

    # Change the 'train_set_path' according to the location where your 'train_set.csv' is
    train_set_path = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\datasets\univ_patras\GSL_split\GSL_split\GSL_isolated\train_set.csv'

    # Change the 'landm_csvs_path' according to the directory where your samples of landmarks csvs are
    train_landm_csvs_path = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\datasets\univ_patras\GSL_split\GSL_split\GSL_isolated\train_points_data_num_frames_10_zero'

    # Change the 'val_set_path' according to the location where your 'val_set.csv' is
    val_set_path = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\datasets\univ_patras\GSL_split\GSL_split\GSL_isolated\val_set.csv'

    # Change the 'val_landm_csvs_path' according to the directory where your validation samples of landmarks csvs are
    val_landm_csvs_path = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\datasets\univ_patras\GSL_split\GSL_split\GSL_isolated\val_points_data_num_frames_10_zero'

    # Change the 'test_set_path' according to the location where your 'test_set.csv' is
    test_set_path = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\datasets\univ_patras\GSL_split\GSL_split\GSL_isolated\test_set.csv'

    # Change the 'val_landm_csvs_path' according to the directory where your testing samples of landmarks csvs are
    test_landm_csvs_path = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\datasets\univ_patras\GSL_split\GSL_split\GSL_isolated\test_points_data_num_frames_10_zero'

    #Change this value to 'True' or 'False' if you want to perform perform parameters tuning or no, respectively. If no you can provide a dictionary with parameters above
    flag_tuning = False
    selected_parameters = {"C": 0.1, "kernel": "linear"} #{"C": 1000, "kernel": "linear"} #{'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}

    #Define the directory to save the model after training
    dir_to_save_model = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\temp_saved_models'
    if not os.path.exists(dir_to_save_model):
        os.makedirs(dir_to_save_model)

    #Define class_weight parameter for SVM
    clas_weight = None #'balanced' #Choose the value 'balanced' to use the heuristic of sklearn where weights are computed based on
                             #the formula 'n_samples / (n_classes * np.bincount(y))'
                             #If you dont want a loss weighting choose the value None.

    X, y, y_numbers, indiv_words = dataset_creation(train_set_path, train_landm_csvs_path)
    X_val, y_val, y_numbers_val, _ = dataset_creation(val_set_path, val_landm_csvs_path)
    X_test, y_test, y_numbers_test, _ = dataset_creation(test_set_path, test_landm_csvs_path)

    scaler = StandardScaler()
    scaler.fit(X)
    X_transformed = scaler.transform(X)
    X_val_transformed = scaler.transform(X_val)
    X_test_transformed = scaler.transform(X_test)


    if flag_tuning:

        # Tune SVM parameters
        params = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'C': [0.1, 1, 10, 100, 1000]},
                  {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}]
        clf = GridSearchCV(SVC(class_weight=clas_weight), params, scoring='f1_macro', n_jobs=8, verbose=3)

    else:
        clf = SVC(**selected_parameters, class_weight=clas_weight)

    clf.fit(X_transformed, y)

    # Path to save results
    path_results = r'C:\Users\georg\Documents\mine\εργασιες\μεταπτυχιακό\2ο εξάμηνο\Μηχανική Μάθηση σε Πολυμεσικά Δεδομένα\1η εργασία\logs'
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    # Txt file path for results of each epoch
    path_txt = path_results + '/epochs_results.txt'

    if flag_tuning:
        res_out1 = "Best parameters found to be:"
        print(res_out1)

        best_par = clf.best_params_
        print(best_par)

    else:
        res_out1 = "Best parameters were set to be:"
        print(res_out1)

        best_par = selected_parameters
        print(best_par)


    if flag_tuning:

        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        mean_std_params_out = ""
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):

            temp_res = "\n%0.3f (+/-%0.03f) for %r" % (mean, std*2, params)
            mean_std_params_out += temp_res
            print(temp_res)


        # Save parameters definition
        with open(path_txt, 'w') as f:
            f.write(res_out1)
            f.write(json.dumps(best_par))
            f.write(mean_std_params_out)

    else:

        # Save parameters definition
        with open(path_txt, 'w') as f:
            f.write(res_out1)
            f.write(json.dumps(best_par))



    #Save the model
    filename = 'svm_model.sav'
    pickle.dump(clf, open(dir_to_save_model + "/" + filename, 'wb'))

    #Make predictions for the validation set using the best parameters found
    y_val_preds = clf.predict(X_val_transformed)

    #Get classification report for validation set
    clas_report = classification_report(y_val, y_val_preds, target_names=indiv_words, labels=indiv_words, zero_division=0)

    prnt_out = '\nClassification report for validation set:\n'
    print(prnt_out)
    print(clas_report)
    with io.open(path_txt, 'a', encoding="utf-8") as ff:
        ff.write(prnt_out)
        ff.write(clas_report)


    #Get confusion matrix for validation set
    cf_matrix = confusion_matrix(y_val, y_val_preds, labels=indiv_words)

    prnt_out = '\nConfusion matrix:\n'
    print(prnt_out)
    print(cf_matrix)
    with open(path_txt, 'a') as ff:
        ff.write(prnt_out)
        ff.write(np.array2string(cf_matrix))

    # Plot confusion matrix for validation set
    df_cm = pd.DataFrame(cf_matrix, index=indiv_words, columns=indiv_words)
    plt.figure(figsize=(12, 10))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(path_results + "/Confusion_matrix_val.png")
    plt.show()


    # Make predictions for the test set using the best parameters found
    y_test_preds = clf.predict(X_test_transformed)

    # Get classification report for test set
    clas_report = classification_report(y_test, y_test_preds, target_names=indiv_words, labels=indiv_words,
                                        zero_division=0)

    prnt_out = '\nClassification report for test set:\n'
    print(prnt_out)
    print(clas_report)
    with io.open(path_txt, 'a', encoding="utf-8") as ff:
        ff.write(prnt_out)
        ff.write(clas_report)

    # Get confusion matrix for test set
    cf_matrix = confusion_matrix(y_test, y_test_preds, labels=indiv_words)

    prnt_out = '\nConfusion matrix for test set:\n'
    print(prnt_out)
    print(cf_matrix)
    with open(path_txt, 'a') as ff:
        ff.write(prnt_out)
        ff.write(np.array2string(cf_matrix))

    # Plot confusion matrix for test set
    df_cm = pd.DataFrame(cf_matrix, index=indiv_words, columns=indiv_words)
    plt.figure(figsize=(12, 10))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(path_results + "/Confusion_matrix_test.png")
    plt.show()


    #####Plot roc_curve#####

    # Binarize labels of train and validation set
    """y_bin = label_binarize(y, classes=indiv_words)
    y_val_bin = label_binarize(y_val, classes=indiv_words)

    # fit SVM model with oneVsRestClassifier
    SVM_roc = OneVsRestClassifier(SVC(**best_par, probability=True))

    # Get the scores using decision_function
    score_SVM = SVM_roc.fit(X_transformed, y_bin).decision_function(X_val_transformed)

    # Compute false positive and true positive rates and construct multi-class ROC curves for SVM
    fpr_SVM = {}
    tpr_SVM = {}
    thresh_SVM = {}
    roc_auc_SVM = dict()
    class_list_SVM = dict()
    for i, clas in enumerate(indiv_words):  # For each class compute false positive and true positive rate
        fpr_SVM[i], tpr_SVM[i], _ = roc_curve(y_val_bin[:, i], score_SVM[:, i])
        class_list_SVM[i] = clas
        roc_auc_SVM[i] = auc(fpr_SVM[i], tpr_SVM[i])

    # Plot ROC curve
    plt.figure()
    for i, clas in enumerate(indiv_words):
        plt.plot(fpr_SVM[i], tpr_SVM[i], label='ROC curve of class {0} (area = {1:0.2f})'''.format(clas, roc_auc_SVM[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC curve for SVM')
    plt.legend(loc="lower right")
    plt.savefig(path_results + "/roc_curve.png")
    plt.show()"""
