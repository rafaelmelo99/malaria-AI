Using Convolutional Neural Network (CNN) to detect Malaria

This CNN detects cells with Malaria. 

The database was collected from the URL: "https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria"

TO TRAIN THE CNN

There was a previous model on folder "output", but you can train it again using the follow command:

python3 cnn-malaria.py -d path_to_database -m path_to_save_model.model

To make a predition, type:

python3 predict.py -i path_to_image.png -m path_to_model.model
