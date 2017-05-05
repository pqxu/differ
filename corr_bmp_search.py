#coding=utf-8
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import gzip as gz, cPickle as pkl
import numpy as np
from PIL import Image
from scipy import signal
from numpy import unravel_index
import function as func

k = 4
m = 1
size = 477
center = (size-1)/2
resize = 513
recenter = (resize-1)/2
r = 225
r_resize = 242

circle477 = func.create_circle(size,r)
circle513 = func.create_circle(resize,r)
circle513_resize = func.create_circle(resize,r_resize)
circle = func.to_Variable(circle477)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(m, 1, 13 , 1 , 0 , 3) # m input image channel
    def forward(self, x , y):
        x = self.conv1(x)
        return x*(-y+1)


if __name__ == '__main__':
  #Set up network
  net = Net()
  for run in np.arange(1,239-k):
    data = func.load(run,run+2*k+m,size)
    rate = 0.00005
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=rate, momentum=0.9)
    #prepare input and label
    input = func.input_change(data[0:m,:,:] , resize , circle477 ,circle513 , r)
    label = data[m+k-1,:,:]
    mov_array =func.get_move(input,label)
    input = func.do_move(input,mov_array,circle513_resize)
    label = func.to_Variable(label)
    base1 = criterion(func.to_Variable(data[m-1,:,:]), label).data.numpy()[0]
    base2 = criterion(func.to_Variable(input[-1,recenter-center:recenter+center+1,
                                            recenter-center:recenter+center+1]), label).data.numpy()[0]
    base0 = base2
    print 'base1 = '+ str(base1)
    print 'base2 = '+ str(base2)
    print 'base2/base1 = '+str(base2/base1)
    base = base2
    # func.save_Variable(func.to_Variable(input[-1,recenter-center:recenter+center+1,
    #                                         recenter-center:recenter+center+1]),'picture/'+str(run+m+k-1)+'_corr_from_'+str(run+m-1)+'.bmp')

    input = func.to_Variable(input)
    label = label

    pred_input = func.do_move(func.input_change(data[k:m+k,:,:] , resize , circle477 ,circle513 , r),mov_array,circle513_resize)
    pred_label = func.to_Variable(data[-1,:,:])
    pred_base1 = criterion(label , pred_label).data.numpy()[0]
    pred_base2 = criterion(func.to_Variable(pred_input[-1,recenter-center:recenter+center+1,
                                            recenter-center:recenter+center+1]), pred_label).data.numpy()[0]
    pred_input = func.to_Variable(pred_input)

    #train the network
    for epoch in range(1000):
      optimizer.zero_grad()
      o = net(input, circle)
      loss = criterion(o, label)
      loss.backward()
      optimizer.step()

      print (loss.data.numpy()[0])/base0

      pred = net(pred_input, circle)
      pred_MSE = criterion(pred, pred_label).data.numpy()[0]
      print 'pred:'+str(pred_MSE/pred_base2)

      if (loss.data.numpy()[0]-base)/base0> 100:
        rate = rate / 3.33333
        optimizer = torch.optim.SGD(net.parameters(), lr=rate, momentum=0.9)
        print '#####################'
        print (loss.data.numpy()[0]-base)/base0
        print (loss.data.numpy()[0])/base0
        print '#####################'
      if abs((loss.data.numpy()[0]-base)/base0)< 0.00004:
        break
      base = loss.data.numpy()[0]

    func.save_Variable(net(input, circle) , 'picture_'+str(k)+'/'+str(run+m+k-1)+'_train.bmp')
    pred = net(pred_input, circle)
    pred_MSE = criterion(pred, pred_label).data.numpy()[0]

    print 'pred_(base2/base1) = '+ str(pred_base2/pred_base1)
    print 'pred_(MSE/base1) = '+ str(pred_MSE/pred_base1)
    print 'pred_(MSE/base2) = '+ str(pred_MSE/pred_base2)
    func.save_Variable(pred, 'picture_'+str(k)+'/'+str(run+m+2*k-1)+'_pred.bmp')
  #   for sa in range(m):
  #     core = list(net.parameters())[0].data.numpy()[0,sa,:,:]
  #     img = Image.fromarray((((core-core.min())/(core.max()-core.min()))*255).astype(np.uint8))
  #     img.save('core/'+str(run+m+k-1)+'_from_' + str(run+sa) + '.bmp')
