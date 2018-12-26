
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import h5py


# In[32]:


cifar_data = h5py.File('CIFAR10.hdf5','r')
x_train = np.float32(np.array(cifar_data['X_train']))
y_train = np.int32(np.array(cifar_data['Y_train']))
x_test = np.float32(np.array(cifar_data['X_test']))
y_test = np.int32(np.array(cifar_data['Y_test']))
cifar_data.close()


# In[33]:


class CIFARModel(nn.Module):
    def __init__(self):
        super(CIFARModel, self).__init__()
        self.conv1 = nn.Conv2d(3,64,4,stride=1,padding=2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout2d()
        self.conv2 = nn.Conv2d(64,64,4,stride=1,padding=2)
        self.conv3 = nn.Conv2d(64,64,4,stride=1,padding=2)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,64,4,stride=1,padding=2)
        self.conv5 = nn.Conv2d(64,64,4,stride=1,padding=2)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64,64,3)
        self.conv7 = nn.Conv2d(64,64,3)
        self.conv7_bn = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64,64,3)
        self.conv8_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(1024,500)
        self.fc2 = nn.Linear(500,500)
        self.fc3 = nn.Linear(500,10)
        
    def forward(self,x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.max_pool2d(F.relu(self.conv2(x)),2,stride=2)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.max_pool2d(F.relu(self.conv4(x)),2,stride=2)
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = self.dropout(F.relu(self.conv6(x)))
        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = self.dropout(F.relu(self.conv8_bn(self.conv8(x))))
        x = x.view(-1,1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x,dim=1)
        return x
    


# In[34]:


model = CIFARModel()
model.cuda()

#Adam Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 100
num_epochs = 100
len_train = len(y_train)
model.train()
train_loss = []

for epoch in range(num_epochs):
    I_permutation = np.random.permutation(len_train)
    x_train = x_train[I_permutation,:]
    y_train = y_train[I_permutation]
    train_accu = []
    if(epoch>10):
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                    if(state['step']>=1024):
                        state['step'] = 1000
    optimizer.step()
    
    for i in range(0, len_train, batch_size):
        x_train_batch = torch.FloatTensor( x_train[i:i+batch_size,:] )
        y_train_batch = torch.LongTensor( y_train[i:i+batch_size] )
        data, target = Variable(x_train_batch).cuda(), Variable(y_train_batch).cuda()
        
        optimizer.zero_grad()
        
        output = model(data)
        
        loss = F.nll_loss(output, target)
        
        loss.backward()
        train_loss.append(loss.data[0])

        optimizer.step()

        prediction = output.data.max(1)[1] 
        accuracy = (float(prediction.eq(target.data).sum())/float(batch_size))*100.0
        train_accu.append(accuracy)
    accuracy_epoch = np.mean(train_accu)
    print("Epoch:"+str(epoch+1)" Accuracy:"+str(accuracy_epoch)" Loss:"+str(loss))

    
model.eval()
test_accu = []

for i in range(0, len(y_test), batch_size):
    x_test_batch = torch.FloatTensor(x_test[i:i+batch_size,:])
    y_test_batch = torch.LongTensor(y_test[i:i+batch_size])
    data, target = Variable(x_test_batch).cuda(), Variable(y_test_batch).cuda()
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    prediction = output.data.max(1)[1] 
    accuracy = (float(prediction.eq(target.data).sum()) /float(batch_size))*100.0
    test_accu.append(accuracy)
accuracy_test = np.mean(test_accu)
print("Test accuracy:"+str(accuracy_test))

