<h1>Using Convolutional Neural Network (CNN) to detect Malaria</h1>

<i>This CNN detects cells with Malaria. </i>

The database was collected from the URL: "https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria"

<h3>TO TRAIN THE CNN</h3>

There was a previous model on folder "output", but you can train it again using the follow command:

<i>python3 cnn-malaria.py -d path_to_database -m path_to_save_model.model</i>

To make a prediction, type:

<i>python3 predict.py -i path_to_image.png -m path_to_model.model</i>

<h3>Requires the following libraries</h3>
<ul>
  <li> rgb2gray(from skimage.color)</li>
  <li> train_test_split(from sklearn.model_selection)</li>
  <li> LabelBinarizer(from sklearn.preprocessing)</li>
  <li> keras(from tensorflow)</li>
  <li> paths(from imutils)</li>
  <li>numpy</li>
  <li>OpenCV</li>
  <li>random</li>
  <li>os</li>
  <li>argparse</li>
 </ul>
  <h3> References to make this CNN
<ul>
  <li>https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5</li>
  <li>https://machinelearningmastery.com</li>
</ul>
