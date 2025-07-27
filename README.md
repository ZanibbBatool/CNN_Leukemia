# CNN_Leukemia
Leukemia Image Detection model using EfficentNetV2-S architecture and C-NMC Leukemia dataset.
The dataset was gathered from Kaggle and preprocessed locally. The dataset folder structure was reformatted and the validation set was removed due to missing 
values. The training data is split into 80:20 ratio for training and validation using 5 fold cross validation. 

Image preprocessing techniques applied to the dataset
include: image resizing (128x128), Contrast Limited Adaptive Histogram Equalization, median blur filter, data augmentation, rotation,
width and height shift, zoom and horizontal flipping

Optimization techniques and hyperparameters : Synthetic General Adversial Network to balance classes, Binary Crossentropy loss function, Adam optimizer,
                                               20 epoch, 16 batch size, ReduceLROnPlateau, Early Stopping, Startified 5 fold training, Resampling,
                                               Shuffling, EfficientNetV2-S pretrained weights
User Interface - Simple Flask webpage that accepts blood smear images, inputs it to the model and sends back a prediction and confidence level to UI
