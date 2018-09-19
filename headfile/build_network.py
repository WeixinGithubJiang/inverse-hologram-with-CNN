
# coding: utf-8

# In[3]:


import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F


class TEST_NET(nn.Module):
    def __init__(self,NUM_INPUT_CHANNEL):
        super(TEST_NET, self).__init__()
        self.conv1 = nn.Conv2d(NUM_INPUT_CHANNEL, 4,kernel_size=9, stride=1,padding=4)  
        self.conv2 = nn.Conv2d(4, 8,kernel_size=9, stride=1,padding=4)  
        self.conv3 = nn.Conv2d(8, 16,kernel_size=7, stride=1,padding=3)  
        self.conv4 = nn.Conv2d(16, 32,kernel_size=7, stride=1,padding=3)
        self.conv5 = nn.Conv2d(32, 64,kernel_size=5, stride=1,padding=2)  
        self.conv6 = nn.Conv2d(64, 128,kernel_size=5, stride=1,padding=2)  
        self.conv7 = nn.Conv2d(128, 256,kernel_size=3, stride=1,padding=1)  
        self.conv8 = nn.Conv2d(256, 512,kernel_size=3, stride=1,padding=1)  
        self.conv9 = nn.Conv2d(512, 256,kernel_size=3, stride=1,padding=1)
        self.conv10 = nn.Conv2d(256, 128,kernel_size=3, stride=1,padding=1)  
        self.conv11 = nn.Conv2d(128, 64,kernel_size=5, stride=1,padding=2)  
        self.conv12 = nn.Conv2d(64, 32,kernel_size=5, stride=1,padding=2)
        self.conv13 = nn.Conv2d(32, 16,kernel_size=7, stride=1,padding=3)
        self.conv14 = nn.Conv2d(16, 8,kernel_size=7, stride=1,padding=3)
        self.conv15 = nn.Conv2d(8, 4,kernel_size=9, stride=1,padding=4)
        self.conv16 = nn.Conv2d(4, 1,kernel_size=9, stride=1,padding=4)
        self.relu = nn.ReLU()

    def forward(self, x0):
        x1 = self.relu(self.conv1(x0))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x6 = self.relu(self.conv6(x5))
        x7 = self.relu(self.conv7(x6))
        x8 = self.relu(self.conv8(x7))
        x9 = self.relu(self.conv9(x8))+x7
        x10 = self.relu(self.conv10(x9))+x6
        x11 = self.relu(self.conv11(x10))+x5
        x12 = self.relu(self.conv12(x11))+x4
        x13 = self.relu(self.conv13(x12))+x3
        x14 = self.relu(self.conv14(x13))+x2
        x15 = self.relu(self.conv15(x14))+x1
        out = self.conv16(x15)
        return out

class TEST_NET2(nn.Module):
    def __init__(self,NUM_INPUT_CHANNEL):
        super(TEST_NET2, self).__init__()
        self.conv1 = nn.Conv2d(NUM_INPUT_CHANNEL, 32,kernel_size=9, stride=1,padding=4)  
#         self.conv2 = nn.Conv2d(4, 8,kernel_size=9, stride=1,padding=4)  
#         self.conv3 = nn.Conv2d(8, 16,kernel_size=7, stride=1,padding=3)  
#         self.conv4 = nn.Conv2d(16, 32,kernel_size=7, stride=1,padding=3)
        self.conv5 = nn.Conv2d(32, 128,kernel_size=5, stride=1,padding=2)  
#         self.conv6 = nn.Conv2d(64, 128,kernel_size=5, stride=1,padding=2)  
        self.conv7 = nn.Conv2d(128, 256,kernel_size=3, stride=1,padding=1)  
#         self.conv8 = nn.Conv2d(256, 512,kernel_size=3, stride=1,padding=1)  
#         self.conv9 = nn.Conv2d(512, 256,kernel_size=3, stride=1,padding=1)
        self.conv10 = nn.Conv2d(256, 128,kernel_size=3, stride=1,padding=1)  
        self.conv11 = nn.Conv2d(128, 64,kernel_size=5, stride=1,padding=2)  
        self.conv12 = nn.Conv2d(64, 32,kernel_size=5, stride=1,padding=2)
        self.conv13 = nn.Conv2d(32, 8,kernel_size=7, stride=1,padding=3)
#         self.conv14 = nn.Conv2d(16, 8,kernel_size=7, stride=1,padding=3)
#         self.conv15 = nn.Conv2d(8, 4,kernel_size=9, stride=1,padding=4)
        self.conv16 = nn.Conv2d(8, 1,kernel_size=9, stride=1,padding=4)
        self.relu = nn.ReLU()

    def forward(self, x0):
        x1 = self.relu(self.conv1(x0))
#         x2 = self.relu(self.conv2(x1))
#         x3 = self.relu(self.conv3(x2))
#         x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x1))
#         x6 = self.relu(self.conv6(x5))
        x7 = self.relu(self.conv7(x5))
#         x8 = self.relu(self.conv8(x7))
#         x9 = self.relu(self.conv9(x8))+x7
        x10 = self.relu(self.conv10(x7))+x5
        x11 = self.relu(self.conv11(x10))
        x12 = self.relu(self.conv12(x11))+x1
        x13 = self.relu(self.conv13(x12))
#         x14 = self.relu(self.conv14(x13))+x2
#         x15 = self.relu(self.conv15(x14))+x1
        out = self.conv16(x13)
        return out
    
# # simple 4-layer FCN
# class FULLY_CONV_NET(nn.Module):
#     def __init__(self,num_input_channel):
#         super(FULLY_CONV_NET, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(num_input_channel, 100,kernel_size=5, stride=1,padding=2),  
#             nn.ReLU(inplace=True),
#             nn.Conv2d(100, 50,kernel_size=7, stride=1,padding=3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(50, 10,kernel_size=7, stride=1,padding=3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(10, 1,kernel_size=3, stride=1,padding=1),
#         )

#     def forward(self, x):
#         out = self.features(x)
#         return out



# # simple 4-layer FCN
# class TEST_NET_1(nn.Module):
#     def __init__(self,num_input_channel):
#         super(TEST_NET_1, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(num_input_channel, 64, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 128, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 64, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 16, kernel_size=7, padding=3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 1, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1, 1, kernel_size=1),
#         )

#     def forward(self, x):
#         out = self.features(x)
#         return out

    

