#coding=utf-8
import torch
from torch.autograd import Variable
import numpy as np
from numpy import unravel_index
from PIL import Image
from scipy import signal
import gzip as gz, cPickle as pkl


def create_circle(size,r):
  out = np.zeros((size,size),dtype = np.uint8)
  center = (size-1)/2
  for i in range(size):
    for j in range(size):
      if np.sqrt((i-center)**2+(j-center)**2)>r:
        out[i,j] = 1
  return out


def load(i,j,size):  #nparray load num times
  nparray = np.zeros((j-i,size,size),dtype = np.float32)
  for num in range(j-i):
    data = Image.open('picture/'+str(num+i)+'.bmp')
    nparray[num,:,:] = np.asarray(data).astype(np.float32)
  return nparray


def input_change(nparray , resize , circle ,circle_big , r):   #nparray are 3D array
  channel = nparray.shape[0]
  size = nparray.shape[1]
  center = (size - 1)/2
  recenter = (resize - 1)/2
  input = np.zeros((channel,resize,resize),dtype = np.float32)
  for j in range(channel):
    a = Image.fromarray(nparray[j,:,:])
    b = a.resize((resize,resize),Image.ANTIALIAS)
    a = np.asarray(a)
    a = a * (-circle+1)
    c = np.zeros((resize,resize),dtype = np.uint8)
    c[recenter-center:recenter+center+1,recenter-center:recenter+center+1] = a
    b = np.asarray(b)
    b = b * circle_big
    input[j,:,:] = b + c
  return input


def to_Variable(nparray):   #from number num to Variable
  if nparray.ndim == 2:
    out = np.array([np.array([nparray.astype(np.float32)])])
    out = Variable(torch.from_numpy(out))
  if nparray.ndim == 3:
    out = np.array([nparray.astype(np.float32)])
    out = Variable(torch.from_numpy(out))
  return out

def save_Variable(Var,save_string):
    Var = np.round(Var.data.numpy()[0,0,:,:])
    Var[Var<0] = 0
    Var[Var>255] = 255
    img = Image.fromarray(Var.astype(np.uint8))
    img.save(save_string)
    print 'save:' + save_string




def get_move(data,label):  #get move array
# data:3D np array , label:2D np array
  center = (data.shape[1]-1)/2
  channel = data.shape[0]
  out = np.zeros((2,channel),dtype = np.int32)
  for j in range(channel):
    temp = signal.fftconvolve(data[j,:,:], label[::-1, ::-1] ,'same')
    x, y = unravel_index(temp.argmax(), temp.shape)
    x = x - center   #向上移动的量
    y = y - center   #向左移动的量
    out[0,j] = x
    out[1,j] = y
  return out

def do_move(data,mov_array,circle_resize):  #move the pixels in inputs
# data:3D np array
  channel = data.shape[0]
  size = data.shape[1]
  out = np.zeros((channel,size,size),dtype = np.float32)
  for j in range(channel):
    temp = np.zeros((size,size),dtype = np.float32)
    x = mov_array[0,j]
    y = mov_array[1,j]
    if x>=0 and y>=0:
      temp[:size-x,:size-y] = data[j,x:size,y:size]
    if x<0 and y<0:
      temp[-x:size,-y:size] = data[j,:size+x,:size+y]
    if x<0 and y>=0:
      temp[-x:size,:size-y] = data[j,:size+x,y:size]
    if x>=0 and y<0:
      temp[:size-x,-y:size] = data[j,x:size,:size+y]
    out[j,:,:] = temp*(-circle_resize+1)
  return out
