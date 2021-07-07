In this file, all python scripts dependencies of this directory are reported as well as how to use them.
All python scripts were created and tested in Windows 10 with python 3.8 version using Pycharm 2021.1.2 software.

The used libraries versions are:

pandas - 1.2.4
torch - 1.8.1
numpy - 1.19.5 
torchvision - 0.9.1
torchinfo - 0.1.2
scikit-learn - 0.24.1
seaborn - 0.11.1
matplotlib - 3.4.1
mediapipe - 0.8.3.1
protobuf - 3.15.8
opencv-python - 4.5.1.48
opencv-contrib-python - 4.5.2.54
Pillow - 8.2.0

######################Dataset##########################################
All the scripts were used along GSLC dataset which can be found here: https://drive.google.com/drive/folders/18ruYi9MULMm1KQtUgdIhN0m-XilRhncg
Only 'GSL_iso_files' are needed from this location.

The structure of these files is:

GSL_split
  |
  |______GSL_isolated
		|
		|_______Greek_isolated
				|
				|_______GSL_isol
					    |
					    |______health1_signer1_rep1_glosses
					    |			|
					    |			|_________________glosses0000
					    | 			|			|
					    |			|			|_______frame_0000.jpg
					    |			|			|_______frame_0001.jpg
					    |			|			.
					    |			|			.
					    |			|			.
					    |			|___________________glosses_0001	
					    |			|			|
					    |			.			|_______frame_0000.jpg
					    |			.			|_______frame_0001.jpg
					    |			.			.
					    |						.
					    |						.
					    |______health1_signer1_rep2_glosses
								|
								|_...
								.
								.
								.

If you want to test another dataset you have to consider this kind of structure.
Train, validation and test set as CSV files can be found here: https://drive.google.com/drive/folders/1XOXsYB6UAfPpsS1VCmmm4wgFZohCjb_e?usp=sharing
as 'train_greek_iso.csv', 'val_greek_iso.csv' and 'test_greek_iso.csv'. These CSVs contain in the first column the paths of the frames in respect with the 
above structure and in the second column the true value of the sign language gesture corresponding to the current path of frames.


##################csv_words_paths_creation_for_n_first_words.py####################
With this file you can create the 'train_set.csv', 'val_set.csv', 'test_set.csv' CSVs files in the appropriate format to use them later in order to 
create the dataset with the bodypoints raw information.

In line 6 you have to specify the type of dataset (train, val, test). First you have to create the train dataset and then the other two.
The CSVs file will be located in the same directory as this python script after its running.

In line 10 you have to specify the path where directory with 'train_greek_iso.csv', 'val_greek_iso.csv' and 'test_greek_iso.csv' is located. 

In line 21 you have to specify the number of the first individual words which appear in 'train_set.csv' and you want to include in your dataset.


##################csv_words_paths_creation_over_n_samples.py####################
With this file you can create 'train_set.csv', 'val_set.csv', 'test_set.csv' CSVs files in the appropriate format to use them later in order to 
create the dataset with the bodypoints raw information.

The discrepancy with the previous script is that with this script you can specify the lower threshold of samples that each individual words should have
in order to be included in your dataset.

Exactly the same as in 'csv_words_paths_creation_for_n_first_words.py' are true (instead of line 21 its 25). 
Additionally, in line 26 you have to specify the lower threshold.


##################points_dataset_creation.py####################
With this file you can create the CSVs which will contain the raw information of bodypoints coordinates which can be fed into the models.

In line 12 you have to specify (True or False) if you want to create the filters mentioned in the corresponding report of this work.

In line 13 you have to specify if you want to sample face points (True or False).

In line 14, if you have chosen True in line 13, you have to specify the rate of sampling (e.g 5).

In line 15 you have to specify the maximum number of frames to be proccessed by Mediapipe. This is related with the sequence length exploited by LSTM or GRU layer.

In line 18 you have to specify the type of dataset you want to create ('train', 'val' or 'test').

In line 21 you have to specify the path of root directory where the frames are located. Based on the above structure, this directory is the 'GSL_isolated'.
In this directory, it should be placed the corresponding 'train_set.csv' or 'val_set.csv' or 'test_set.csv' created by the 'csv_words_paths_creation_for_n_first_words.py' or 'csv_words_paths_creation_over_n_samples.py' scripts. 
The rest of the path structure is considered to be the same as above. If not, you have to specify (in line 27) which is the rest path where root frames directory is located.
Based on the above structure, this rest part of the root directory is 'Greek_isolated/GSL_isol/'. 

After running this script, a folder which will be called like 'train_points_data_num_frames_10_zero' will be located in the directory specified in line 21.

##################points_dataset_creation_angles_magn.py####################
With this file you can create the CSVs which will contain the raw information of bodypoints magnitudes and angles of two consecutive frames which can be fed into the models.

The specifications needed are the same as in points_dataset_creation.py but in different lines (one line shifted downwards). The correspondence of the two files' lines are:

(left line corresponds to 'points_dataset_creation.py' file)
line 12 --> 13
line 13 --> 14
line 14 --> 15
line 15 --> 16
line 18 --> 19
line 21 --> 22
line 27 --> 28
 

##################annotate_img.py####################
With this file you can create the annotated images with the skeleton of body points which then can be fed into the models.

In line 14 you have to specify the type of dataset you want to create ('train', 'val' or 'test').

In lines 19, 20, 21 should be provided the demanding width, height and channels number of the annotated images, respectively.

In line 23 should be provided the directory path where 'train_set.csv' or 'val_set.csv' or 'test_set.csv' is located created by the 'csv_words_paths_creation_for_n_first_words.py' or 'csv_words_paths_creation_over_n_samples.py' scripts.

In line 24 should be provided the root directory where the annotated images will be stored.

In line 26 should be provided the root directory where frames are located. Accordinng to the above structure, this is the directory 'GSL_isol'.

In line 29-36 you can provide (or leave the default) the connections of datapoints you want annotate on the image.


##################model_pytorch.py####################
With this file you can train a deep learning model using the dataset created by the script 'points_dataset_creation.py' or 'points_dataset_creation_angles_magn.py'. 
After the completion of running, two folders called 'logs' and 'temp_saved_models' (by default, but they can be changed).
'logs' folder will contain the results of the training and evaluation proccess. Evaluation proccesing is about both validation and test set.
'temp_saved_models' will contain the model saved after its training.

In line 568 you have to specify the path where 'train_set.csv' (created by the script 'csv_words_paths_creation_for_n_first_words.py' or 'csv_words_paths_creation_over_n_samples.py') is located.

In line 571 you have to specify the path where root directory of training CSV files (created by the script 'points_dataset_creation.py' or 'points_dataset_creation_angles_magn.py') is located. These files should correspond to 'train_set.csv'.

In line 578 you have to specify the path where 'val_set.csv' is located (created by the same script as the 'train_set.csv').

In line 581 you have to specify the path where root directory of validation CSV files is located (created by the same script as the training CSV files). These files should correspond to 'val_set.csv'.

In line 588 you have to specify the path where 'test_set.csv' is located (created by the same script as the 'train_set.csv').

In line 591 you have to specify the path where root directory of test CSV files is located (created by the same script as the training CSV files). These files should correspond to 'test_set.csv'.

In line 606 you can specify the batch size (default is 128).

In line 617 you can specify the number of epochs (default is 500).

In line 626 you can specify the value of learning rate (default is 0.001).

In line 628 you can specify the value of cells (LSTM or GRU) of hidden layers (default is 100).

In line 629 you can specify the number of hidden layers (default is 1).

In lines 640-641 you can specify the type of loss (valid and tested values are 'CrossEntropyLoss' and 'MultiLabelSoftMarginLoss'. Default is 'CrossEntropyLoss'). 

In line 644 you can specify (True or False) if you want weights to be considered in loss function (this is only about 'CrossEntropyLoss').

In lines 657-658 you can specify the type of optimizer (only 'Adam' was tested and it is the default value).

In line 661 you have to specify the path of directory you want to be stored the 'temp_saved_models' folder. 

In line 666 you can specify the patience value of early stopping (default is 10).

In line 669 you can specify the lower difference between the number of epochs specified by 'patience' value (default is 0.0001). Based on this value the stopping of training will be raised.

In line 673 you have to specify the path of directory you want to be stored the 'log' folder.  

In line 127 you can specify (True or False) if you want a 1D CNN layer as the first model layer.

In lines 132, 133, 134, 135, 136 you can specify (if at last you want the CNN layer) the number of CNN filters (default 128), the kernel size (default 5), dilation value (default 1), padding (default 0), stride (default 1), respectively.

In lines 141-144 you can specify (if at last you want the CNN layer) the value of Max pooling kernel size (default is 5), dilation value (default is 1), padding (default is 0), stride (default 5), respectively. 

In line 187 you can specify (if at last you want the CNN layer) the percentage of dropout layer which follows the CNN layer (default is 0.2). 

In lines 193 and 202 you can specify (True or False) if you want LSTM or GRU layers (default is GRU), respectively. Always, the two values should be opposite (one should be True and the other False). 

In lines 199 and 209 you can specify the percentage of dropout layer which follows the LSTM or GRU layer (default is 0.2), respectively.

In line 212 you can specify the number of neurons of the first Fullly Connected layer (default 128). The same value should be specified in line 224.

In line 215 you can specify the percentage of dropout layer which follows the first Fully Connected layer (default is 0.2).


##################model_pytorch_images.py####################
This file carries out the same operation as 'model_pytorch.py' (produces the same outputs folders 'logs' and 'temp_saved_models') with the discrepancy that the implemented models (with 2D CNN) take as input the annotated images created by the 'annotate_img.py' script.
In this case, the CNN layer is not optional.

The order and number of lines to be specified is a little bit different in relation with the 'model_pytorch.py'. The correspondance of lines is illustrated below (also some default values are different and these are referred):

(left line corresponds to 'model_pytorch.py' file)
line 568 --> 613
line 571 --> 616
line 578 --> 623
line 581 --> 626
line 588 --> 633
line 591 --> 636
line 606 --> 651 (default batch size is 64)
line 617 --> 662
line 626 --> 673
line 628 --> 674 (default number of LSTM/GRU cells is 20)
line 629 --> 675
line 640-641 --> 684-685
line 644 - 688
line 657-658 --> 701-702
line 661 --> 705
line 666 --> 710
line 669 --> 713
line 673 --> 717
line 127 --> 173
line 132, 133, 134, 135, 136 ---> 177 (default filters of CNN layer is 16), 178, 179, 180, 181
line 141 - 144 --> 187 - 190
line 187 - 238
line 193 --> 244
line 202 --> 253
line 199 --> 250
line 209 --> 260
line 212 --> 263
line 224 --> 275
line 215 --> 266 


##################svm_model.py####################
With this file you can train an SVM classifier using the dataset created by the script 'points_dataset_creation.py' or 'points_dataset_creation_angles_magn.py'. 
The same output folders as in file 'model_pytorch.py' are produced after completing its running.

In line 97 you have to specify the path where 'train_set.csv' (created by the script 'csv_words_paths_creation_for_n_first_words.py' or 'csv_words_paths_creation_over_n_samples.py') is located.

In line 100 you have to specify the path where root directory of training CSV files (created by the script 'points_dataset_creation.py' or 'points_dataset_creation_angles_magn.py') is located. These files should correspond to 'train_set.csv'.

In line 103 you have to specify the path where 'val_set.csv' is located (created by the same script as the 'train_set.csv').

In line 106 you have to specify the path where root directory of validation CSV files is located (created by the same script as the training CSV files). These files should correspond to 'val_set.csv'.

In line 109 you have to specify the path where 'test_set.csv' is located (created by the same script as the 'train_set.csv').

In line 112 you have to specify the path where root directory of test CSV files is located (created by the same script as the training CSV files). These files should correspond to 'test_set.csv'.

In line 115 you have to specify (True or False) if you want to perform a greedy search for SVM parameters tuning.

In line 116 (if you dont want a greedy search) you can specify the SVM parameters (default is {"C": 0.1, "kernel": "linear"}).

In line 119 you have to specify the path of directory you want to be stored the 'temp_saved_models' folder.

In line 124 you have to specify if you want the loss function to take into account the weight of each class (valid values are 'balanced' and None. Default is None).

In line 152 you have to specify the path of directory you want to be stored the 'log' folder. 

You can uncomment the lines 272-305 if you want to be returned the ROC-AUC curves with the rest results (but it takes much more time).



##################words_distribution.py####################
With this file you can get the histogram consists of the number of samples for each word for a specific dataset created by the script 'points_dataset_creation.py' or 'points_dataset_creation_angles_magn.py'. 

In line 8 you have to specify the type of dataset for which you want to create the histogram (the valid values are 'train', 'val' and 'test'. Default is 'train').

In line 10 you have to specify the type of CSV from which you want to create the histogram. If the selected value is 'from_initial_data' 
the initial CSV file ('train_greek_iso.csv'which is provided by the google drive link above) will be used. For this reason the selected value in line 8 should be 'train'. 
Else, if the selected value is 'from_modified_data' (which is the default) one of the created CSV files ('train_set.csv' or 'val_set.csv' or 'test_set.csv' created by the script 'points_dataset_creation.py' or 'points_dataset_creation_angles_magn.py') will be used according to your choice in line 8. 

In line 13 you have to specify the path where the CSV of dataset is located (the demanding CSV depends on the selected type of dataset (selected in line 8) and the type of CSV (selected in line 10)).


##################demonstration_demo.py####################
With this file you can have a visualized demo where the initial video frames and the annotated by Mediapipe video frames are displayed 
as well as the predicted class from the deep learning model (model with Architecture 1 which is described in the corresponding report of 
this work) and the actual class for a particular set of video frames. Deep learning model performs a prediction after the first 10 frames 
in sliding windows (e.g a prediction for the frames 1-10, a prediction for the frames 2-11, a prediction for the frames 3-12 etc).

In line 259 you have to specify which sample (default is 3) to be used for the visualized demo.

In line 261 you have to define a list with the partial paths of the samples you want to perform the demo (based on the above structure one example of this partial path is 'health1_signer1_rep1_glosses/glosses0000'). 

In line 274 you have to define a list with the actual classes of the samples that you defined in line 261.
 
In line 275 you have to specify the path where the root directory of the frames are located (based on the above structure this directory is 'GSL_isol').

In line 277 you have to specify the path where the pretrained model is located. You can find this model (as 'model.pth') here: https://drive.google.com/drive/folders/1XOXsYB6UAfPpsS1VCmmm4wgFZohCjb_e (is the link which provided above).
The model architecture defined in this script is alligned with the provided model. If you want to use your own pretrained model you have to define again the whole archtecture in this script to match with your model.
In addition, if you want to use a different model, you have to specify a list (in line 287) with the individual words that the model learnt, in the same order as the output of the model, and the number number of these individual words (in line 12).

In line 278 you have to define where is located the file of the font type to be used for the vizualization of the predicted and actual class. 
You can find the one which was used, in the above google drive link as 'Ubuntu-R.ttf'. If you want to use a different type of font you have to find the corresponding file and provide also its name in line 511.


##################demonstration_live.py####################
With this file you can have a visualized demo as of the script 'demonstration_demo.py', but instead of using samples from the dataset, frames of your webcam will be used.
In this case, only the predictions of model are displayed as there are not ground truth labels.

In line 268 you have to specify the path where the pretrained model is located (the model should be the same as of the script 'demonstration_demo.py'). 

In line 269 you have to specify the path where is located the file of the font type to be used for the vizualization of the predicted class (here it is valid that it is valid for the script 'demonstration_demo.py').