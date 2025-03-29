# -Image-Classification-using-CNNs-Deep-Learning-

 This code is a complete end-to-end for Image Classification using Convolutional Neural Networks (CNNs). It includes building a custom CNN from scratch and fine-tuning a pre-trained model to improve accuracy. Let's break it down step by step:

 1. Setting Up the Environment
Installing the required libraries:
•	TensorFlow: A Deep learning framework for building and training CNNs.
•	NumPy: For handling arrays and numerical operations.
•	Matplotlib: For plotting graphs to visualize training progress.

 2. Importing Required Libraries
•	Imported necessary modules for building and training deep learning models using TensorFlow/Keras.
•	Importing VGG16 for transfer learning.
•	Importing ImageDataGenerator for data augmentation (though not used here).

3. Loading & Preprocessing the CIFAR-10 Dataset
Loading the CIFAR-10 dataset, which has 60,000 images (50,000 training & 10,000 testing) divided into 10 classes. Normalising pixel values to a range of 0-1 to help the model train better.Defining class names for easy reference.

 4. Building a CNN from Scratch
Architecture Explanation:
•	Convolutional Layers: Detects features using filters (32, 64, 128) of size (3,3).
•	MaxPooling Layers: Downsamples the feature maps by selecting maximum values in a (2,2) block, reducing spatial size.
•	Flatten Layer: Converts the matrix into a single vector for the Dense layer.
•	Dense Layers: Fully connected layers for classification.
•	Output Layer: Uses softmax activation to classify into 10 categories.
•	Optimizer: Adam (adaptive learning rate).
•	Loss Function: Sparse Categorical Crossentropy (suitable for integer labels).

 5. Training the CNN
•	Training the model for 5 epochs using the training dataset.
•	Evaluating performance on the test dataset after each epoch.
•	Returns a history object containing accuracy and loss metrics for plotting.

 6. Evaluating the Model
Evaluating the model's performance on the test data.Printing the final test accuracy.

 7. Fine-Tuning with a Pre-trained Model (VGG16)
•	Transfer Learning: Using pre-trained VGG16 without the top layer (include_top=False).
•	Freezing the VGG16 layers to preserve learned features.
•	Adding custom Flatten & Dense layers for CIFAR-10 classification.
•	Using a smaller learning rate (0.0001) to avoid disrupting pre-trained weights.

 8. Training the Fine-Tuned Model
Training the fine-tuned model for 5 epochs.Compare its performance with the original CNN.

 9. Evaluating the Fine-Tuned Model
Evaluating and printing the final test accuracy of the fine-tuned model.

 10. Plotting Accuracy & Loss Curves
Plots showing Training & Validation Accuracy and Loss for both models. Helps to visualize how well the models are learning and generalising.



