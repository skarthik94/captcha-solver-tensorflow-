# captcha-solver-tensorflow
Solving captcha images using tensorflow

Motivation and Introduction

The primary motivation for this project comes from friend who scraped these images from Amazon while scraping Amazon reviews for a NLP project. As he wasn’t using them, I thought I could make a project out of them. Further motivation for this project comes from how prevalent captcha images are now. Given that we can train a convolutional neural network to recognize individual letters, I thought that I could create a model that solves these captcha images without human intervention.

A captcha is defined as “a program or system intended to distinguish human from machine input, typically as a way of thwarting spam and automated extraction of data from websites.” Captcha images specifically are jumbled up letters or numbers that one has to input to verify that one is human.

For the image below, the input “PNRHXR” would be expected.
 
There are 10000 of these images in the data set.

I used Convolutional Neural Networks for this project. Convolutional Neural Networks are a class of deep neural networks that are most commonly applied to image data. They use a variation of multilayer perceptrons (filters) to identify characteristics within images to identify them.


Methods and Implementation

The first thing that had to be done is that the images had to be split up into 6 individual images, ie. each of the 6 letters. The image above would be split as such:
 
These letters needed to be sorted into separate folders with the folder name being that particular letter. Ex. a folder full of “A” named “A”. This was done through a separate script independent of the notebook with the model in it. This script was able to separate the letters in the image to a decent degree of accuracy. 

This is done through a combination of OpenCV functions and pyTesseract. The images were converted to greyscale. Then they were given some padding and converted to pure black and white. Using the contours of the different letters (continuous blobs of pixels), I was able to isolate the individual letters. The labels were assigned using pyTesseract but the accuracy of this was rather bad. I had to go back into the files and manually sort a large number of the letters into the correct file.

After the pre-processing, I was able to start with the actual model. The first thing that had to be done was to change the images to grey scale and change the size of the images to 20x20. This reduces the amount of memory that would be needed. The model doesn’t necessarily need the full size image. Training on the full size image would have also have taken more time to train and possibly a much bigger network. The labels for the different folders are taken as each image is processed. These labels are one-hot encoded so they can work with the CNN output and so we can understand the output that the model is returning.

The CNN itself has an input layer, 2 hidden layers and 2 max pooling layers. The output layer is fully connected. The architecture of the CNN is a bit beyond me so for this I received some advice from a few friends on how to set it up and what to search for online.

Cross entropy was used to define the loss function for the model and the adam optimizer was used to set the learning rate. Both of these are new concepts to me. I understand however that the adam optimizer is similar to a combination of adagrad and RMSprop.

Before training the model the hyperparameters of the model have to be set namely, the seed, number of epochs and the minibatch size (has to be played around with due to memory, landed on 64 at the end). For each epoch, a function to get minibatches is called with an incremented seed every time. In this way all of the minibatches received will be equal. The cost is printed after every two epochs.

For the test images, a similar process to what happened to preprocess the training images has to be done. The image is converted to grey scale, padded, changed to pure black and white, then the contours are found and the image is split into 6 individual images with a letter in each of them. These letters are passed individually through the model to get the predictions for each letter using the one-hot encoded label that was pickled during training. These are combined to get the final prediction.

All of these steps were run on a 32 GB RAM linux deep learning instance on AWS using jupyter notebook. Getting this set up was a challenge in itself.  Attempting to train even a simple model on my windows machine repeatedly crashed it, requiring a hard reset. I could not find a definitive reason why this was happening.





Experiments 

My experiments with this project were largely figuring out how the code worked, how to organise the code and getting the structure of all the data right. A good chunk of my code was found and repurposed for my own use. Given my lack of experience with image data and neural networks, I felt this was the best approach to getting this project running along with heavily using resources. I did not go about this blindly, only committing to using code after understanding how it was being used and why. One of the bigger issues I had was with memory. Initially I was playing with the code on my windows machine with 16GB RAM but on attempting to use the actual data, the system crashed. Given that the data was nowhere near that large, in my ignorance I figured it must be something in the code. However running the same code on Google colab, it was throwing up errors instead of nothing at all like I was experiencing. I then set up an 8GB AWS instance to run it and it didn’t work then as well but this time I actually got an error message. I experimented with playing around with the minibatch size and image size as well. Changing around the minibatch size didn’t work and reducing the image size reduced the accuracy of the model. After some research, I bumped it up to 16GB but no luck. After a last push to 32GB the code was running bar a few errors here and there.

The architecture of the CNN was something I played around with too. Adding more layers increases the training time substantially with only a slight increase in the accuracy on the training data. Too few layers make the model not perform. Along with the architecture, I also had constant issues with the dimensions of the data, largely to my inexperience with CNNs.

I also wanted to figure out adding a GPU to the mix. Given that the training didn’t take too long to do, it would have been pretty unnecessary. Even so it would have been nice to play around with. I was trying to figure out how to add a GPU to the instance without creating an entirely new instance and adding a GPU during creation. I couldn’t get to happen but I might work on this further during the break. 


Conclusion

You can use neural networks to solve captcha images. The model that I’ve created will only work specifically for the type of captcha it was trained on, that is captchas from the amazon website. It is to be seen if the model can handle other varieties. If there were any lines or dots going through any of the letters, this model would not work due to the contours. Windows machines have an odd relationship with deep learning and it’s probably safer to use linux.


References

https://www.tensorflow.org/tutorials/
https://colab.research.google.com/notebooks/welcome.ipynb#scrollTo=9wi5kfGdhK0R
http://nitin-panwar.github.io/Breaking-CAPTCHAs-using-machine-learning/
http://www.captcha.net/
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_intro/py_intro.html
https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/
https://github.com/madmaze/pytesseract
