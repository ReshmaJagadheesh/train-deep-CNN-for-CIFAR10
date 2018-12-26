# Train-deep-CNN-for-CIFAR10
Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset. 

A deep fully connected Neural network was implemented for training the CIFAR 10 model. The architecture is trained using Adam with a learning rate of 0.001. For CNN, ReLU activation function was used, in the last layer Softmax was used for classification.  The model was trained for 100 epochs and batch size = 100. The CNN achieved an accuracy over 80% when the following architecture was used. 

Convolution layer 1: 64 channels, k = 4,s = 1, P = 2. Batch normalization  
Convolution layer 2: 64 channels, k = 4,s = 1, P = 2. Max Pooling: s = 2, k = 2.  
Convolution layer 3: 64 channels, k = 4,s = 1, P = 2. Batch normalization  
Convolution layer 4: 64 channels, k = 4,s = 1, P = 2. Max Pooling  
Convolution layer 5: 64 channels, k = 4,s = 1, P = 2. Batch normalization  
Convolution layer 6: 64 channels, k = 3,s = 1, P = 0 → Dropout  
Convolution layer 7: 64 channels, k = 3,s = 1, P = 0. Batch normalization  
Convolution layer 8: 64 channels, k = 3,s = 1, P = 0. Batch normalization → Dropout  
Fully connected layer 1: 500 units. ReLU 
Fully connected layer 2: 500 units. Linear → Softmax function 
