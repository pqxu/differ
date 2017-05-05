#coding=utf-8
import numpy as np
from PIL import Image


rain_thresh = 0.07


def evaluater(xp, init, pred, rain, obs):

    frain = rain > rain_thresh
    orain = obs > rain_thresh

    hit = xp.sum(orain * frain)
    falsealarm = xp.sum(frain * (1 - orain))
    missing = xp.sum((1 - frain) * orain)
    trivial = xp.sum((1 - frain) * (1 - orain))

    print 'EVERY:', hit, falsealarm, missing, trivial

    bias = (hit + falsealarm) / (hit + missing + 1e-5)
    csi = hit / (hit + falsealarm + missing + 1e-5)
    tss = (hit * trivial - falsealarm * missing) / ((hit + missing) * (falsealarm + trivial) + 1e-5)

    tmp = ((hit + missing) * (hit + falsealarm) + (missing + trivial) * (falsealarm + trivial)) / (
        hit + falsealarm + missing + trivial + 1e-5)
    hss = (hit + trivial - tmp) / (hit + falsealarm + missing + trivial - tmp + 1e-5)

    pod = hit / (hit + missing + 1e-5)
    far = falsealarm / (hit + falsealarm + 1e-5)

    tmp = ((hit + falsealarm) * (hit + missing)) / (hit + falsealarm + missing + trivial + 1e-5)
    ets = (hit - tmp) / (hit + falsealarm + missing - tmp + 1e-5)

    ratio = xp.mean((obs - pred) * (obs - pred) + 1e-5) / xp.mean((obs - init) * (obs - init) + 1e-5)

    return bias, csi, tss, hss, pod, far, ets, ratio


k = 25
start = 11
over = 50
data = np.zeros((k,8),dtype = np.float32)

if __name__ == '__main__':
  for run in np.arange(start,start+k):
    pre_0 = np.asarray(Image.open('picture_3/'+str(run)+'_pred.bmp'),dtype = np.float32)/256
    pre_1 = np.asarray(Image.open('picture_3/'+str(run+1)+'_pred.bmp'),dtype = np.float32)/256
    pre_2 = np.asarray(Image.open('picture_3/'+str(run+2)+'_pred.bmp'),dtype = np.float32)/256
    pre_3 = np.asarray(Image.open('picture_3/'+str(run+3)+'_pred.bmp'),dtype = np.float32)/256
    pre_4 = np.asarray(Image.open('picture_3/'+str(run+4)+'_pred.bmp'),dtype = np.float32)/256

    obs_0 = np.asarray(Image.open('picture/'+str(run)+'.bmp'),dtype = np.float32)/256
    obs_1 = np.asarray(Image.open('picture/'+str(run+1)+'.bmp'),dtype = np.float32)/256
    obs_2 = np.asarray(Image.open('picture/'+str(run+2)+'.bmp'),dtype = np.float32)/256
    obs_3 = np.asarray(Image.open('picture/'+str(run+3)+'.bmp'),dtype = np.float32)/256
    obs_4 = np.asarray(Image.open('picture/'+str(run+4)+'.bmp'),dtype = np.float32)/256
    init = np.asarray(Image.open('picture/'+str(run-1)+'.bmp'),dtype = np.float32)/256
    pred = pre_0
    rain = np.max([pre_0,pre_1,pre_2,pre_3,pre_4],axis = 0)
    obs = np.max([obs_0,obs_1,obs_2,obs_3,obs_4],axis = 0)
    # img = Image.fromarray(rain*256)
    # img.show()
    data[run-start,:] = evaluater(np, init[over:-over,over:-over], pred[over:-over,over:-over], rain[over:-over,over:-over], obs[over:-over,over:-over])
    # bias, csi, tss, hss, pod, far, ets, ratio = evaluater(np, init, pred, rain, obs)
    # bias, csi, tss, hss, pod, far, ets, ratio = evaluater(np, init, init_corr, init_corr, obs)
    # print '%.2f' %bias, '%.2f' %csi, '%.2f' %tss, '%.2f' %hss, '%.2f' %pod, '%.2f' %far, '%.2f' %ets, '%.2f' %ratio
  print np.mean(data,axis = 0)
  print np.max(data,axis = 0)
