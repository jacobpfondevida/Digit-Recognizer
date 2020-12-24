# Digit-Recognizer
This repository is my exploration of convolutional neural networks for the use of computer vision, and the main library used here is Keras, and more specifically, I practiced using the Sequential model and Conv2D layers. Kaggle's Digit Recognizer competition makes use of the MNIST handwritten digit dataset, and I likewise used it, achieving 95% accuracy with this model. Public notebooks of top competitors and their explanations of how they reached their solutions were an excellent learning tool for me, and I'd like to give them credit as well.

Additionally, I'd like to use this README as a medium through which I can document the important pieces of information I learned in hopes that someone might find something useful in my written realizations.

### Important layers used
The convolutional (Conv2D) layer is perhaps the most important layer for this project, for it is the one that reads in inputted features and creates different metrics for determining what a certain handwritten number is. Essentially, it isolates patterns in a given class (in this case a digit 0-9) and searches through the image to see if it finds any of those distinguishing features within a given kernel. The filters noted in the code are these features and may be, more concretely, something like a dark edge in a certain spot of the picture or a handle on a car.

Pooling (MaxPool2D) and Dropout layers supplement this process of finding distinguishable features by choosing the highest value in a given block and randomly "dropping" certain values, respectively, in order to reduce overfitting to the training dataset.

### Optimizer and annealer
Once the layers -- the architecture -- of the model are set up, we need feedback loops to determine whether the model is optimizing itself correctly. This is where loss (categorical crossentropy) comes into play, measuring the error of what the model classified the training example as and what the example actually is. To make use of this metric, we use a method called gradient descent to adjust the weights (importance) of the aforementioned filters. To make sure we don't overshoot when changing weights, we use an "annealer" which reduces the amount gradient descent changes weights periodically if there is no improvement in classification accuracy.

### Data augmentation
To make the results here more generalizable, I made use of data augmentation -- flipping, rotating, and zooming images so that the model may be better equipped to deal with more varied cases. The idea here is that real-world data may deviate significantly from the dataset in use, and if one manipulates the training examples to a certain degree, the model might be able to overcome these discrepancies.

### Signing off
In all, this project was a good way of solidifying my understanding of convolutional neural networks and Python in general, and I was able to learn several things throughout the implementation and debugging processes. For future applications, it would be interesting to deal with more complex images of non-uniform dimensions and an RGB format, but in the meantime, I feel this is a good project to show the essentials of convolutional neural networks.
