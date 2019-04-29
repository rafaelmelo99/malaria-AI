<h1>Using Convolutional Neural Network (CNN) to detect Malaria</h1>

<i>This CNN detects cells with Malaria. </i>

The database was collected from the URL: "https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria"

<h3>TO TRAIN THE CNN</h3>

There was a previous model on folder "output", but you can train it again using the follow command:

<i>python3 cnn-malaria.py -d path_to_database -m path_to_save_model.model</i>

To make a predition, type:

<i>python3 predict.py -i path_to_image.png -m path_to_model.model</i>
