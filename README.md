# Ch5-Computer-Vision
## **TASK 1**

I have completed a task that involves various image processing operations using several common techniques in computer vision. Here is a summary of what I have done in the task:

1. **Library Introduction and Preparation**: I imported several libraries that will be used, such as NumPy, OpenCV, scikit-image, PyTorch, and Matplotlib. I also defined some utility functions, like `plot` and `apply_clahe_rgb`, that will be used in the task.

2. **Loading the Dataset**: I loaded two images, namely "photo1.jpeg" and "lena.png," which will be used in this task.

3. **Initial Image Processing**: I started by performing some initial image processing operations on the "lena.png" image using OpenCV. This includes changing the color channels from BGR to RGB, converting the color image to a grayscale image, and converting the grayscale image to a binary image using thresholding.

4. **Plotting Histograms**: I displayed histograms of the original image and the processed grayscale image. Histograms provide a visual overview of pixel intensity distributions in the image.

5. **Max Pooling (Scikit-image)**: I applied the max pooling operation to the image using the scikit-image library. Max pooling is an operation that combines multiple pixels into one pixel by selecting the maximum value.

6. **Max Pooling (PyTorch)**: I converted the image into a PyTorch tensor and applied max pooling using PyTorch. I also explained the differences between max pooling in PyTorch and scikit-image.

7. **Min Pooling and Average Pooling**: I applied min pooling and average pooling operations to the image. Min pooling selects the minimum value in a window, while average pooling calculates the average value in a window.

8. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: I applied the CLAHE technique to the image. CLAHE is an image processing technique that enhances local contrast while preserving details in the image.

9. **Saving Processed Images**: I saved the image that was processed using the CLAHE technique in a file with a .png extension.

10. **Questions and Explanations**: I answered some questions related to the differences between max pooling and CLAHE, as well as the advantages of using CLAHE to enhance dark-toned images.

This task provided me with a fundamental understanding of several common image processing techniques used in computer vision. I applied image processing operations, analyzed histograms, and compared the results of various processing techniques.

## **TASK 2**
I have completed a task focused on Transfer Learning using pre-trained models in PyTorch for image classification on the MNIST dataset. Here is a summary of the tasks I have accomplished:

**1. Import Libraries:** I imported various libraries needed for the task, including PyTorch, Torchvision, and other modules. I also set up random seed for reproducibility.

**2. User-defined Functions (UDFs):** I defined several functions used in the task, including functions for checking the number of model parameters, obtaining dataloaders, training the model, and plotting the model's performance.

**3. Model Definition:** I created a `VisionModel` class that allows me to choose between pre-trained models such as ResNet18, DenseNet121, and the Vision Transformer (ViT). I customized the input and output layers of these models to suit the task's requirements, for example, modifying the input layer to accept Grayscale images and the output layer for 10 classes.

**4. Setting the Device:** I set up the device (CPU or CUDA) to be used for model training.

**5. Definition of Hyperparameters:** I specified parameters such as the number of epochs, batch size, and learning rate.

**6. Fetching Train and Validation Dataloaders:** I obtained dataloaders for the training and validation data by sampling a small portion of the MNIST dataset.

**7. Loss Function and Optimizer Selection:** I determined the loss function (Cross-Entropy) and optimizer (Adam) to be used during model training.

**8. Model Training:** I trained the model with the specified number of epochs using the training and validation dataloaders. I followed the training steps involving calculating the loss, backpropagation, and model optimization.

**9. Plotting Model Performance:** After training, I plotted the model's performance, including loss and accuracy graphs during both training and validation.

**10. Retraining Models with Frozen Layers:** I attempted to retrain models with certain layers frozen, specifically "denseblock1" and "denseblock1" along with "denseblock2". I retrained these models and recorded the results.

**11. Evaluating Performance Differences:** I plotted the performance of models retrained with frozen layers and compared them to the original model where all layers can be trained. I also observed differences in accuracy and loss.

**12. Additional Analysis:** I answered questions related to the differences in performance found in models with frozen layers.

**13. Execution Time Measurement:** I recorded the time required to train and validate the models in various scenarios, including when the entire model can be trained and when some layers are frozen.
