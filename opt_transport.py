import sys
import numpy as np
import matplotlib.pylab as plt
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import pickle
from datetime import datetime
from utillc import *
from scipy.stats import norm
import argparse
from joblib import Parallel, delayed
import multiprocessing
from functools import partial

n_bins = 1_000  # nb bins : 7'' pour calculer 20 ot entre 2 dist

# bin positions
x = np.arange(n_bins, dtype=np.float64)
moy, ss = 0, 100
dd = norm.pdf(x, moy, ss)
dd1 = gauss(n_bins, m=moy, s=ss)  # m= mean, s= std

EKON(np.sum(dd), np.sum(dd1))
#plt.plot(x, dd, dd1); plt.show()

nmb_gaussians = 1

def my_gauss(bins, m, s) :
  """
  ma pdf gère mieux les troncatures qui se passent aux bords
  """
  dd = norm.pdf(x, m, s)
  if np.sum(dd) < 0.99 :
    raise Exception("trop pres du bord")
  
  #dd1 = gauss(bins, m=m, s=s)  # m= mean, s= std
  return dd


def ev_(mx, mm, ss) :
  moya1, moya2, moyb1, moyb2 = mm
  stda1, stda2, stdb1, stdb2 = ss
  ma1, ma2, mb1, mb2 = mx
  
  a1 = my_gauss(n_bins, m=moya1 * n_bins, s=stda1 * n_bins)  # m= mean, s= std

  a = a1*ma1
  if nmb_gaussians > 1 :
    a2 = my_gauss(n_bins, m=moya2 * n_bins, s=stda2 * n_bins)
    a += a2*ma2
  b1 = my_gauss(n_bins, m=moyb1 * n_bins, s=stdb1 * n_bins)  # m= mean, s= std
  b = b1 * mb1
  if nmb_gaussians > 1 :
    b2 = my_gauss(n_bins, m=moyb2 * n_bins, s=stdb2 * n_bins)
    b += b2*mb2
  #EKON(np.sum(a), np.sum(b)) 
  # loss matrix
  M = ot.dist(x.reshape((n_bins, 1)), x.reshape((n_bins, 1)))
  M /= M.max()

  #plt.plot([i for i,_ in enumerate(a)], a, b)
  #plt.show()


  G0, cst = ot.emd(a, b, M, log=True,numThreads='max')


  EKON(cst['cost'], np.abs(moya1-moyb1) + (stda1 + stdb1 - 2. * np.sqrt(stda1 * stdb1)))   
  plt.plot([i for i,_ in enumerate(a)], a, b)
  plt.title(str(cst['cost']))
  plt.show()


  
  return (cst['cost'], mx, mm, ss), (a, b, M, G0, cst)

def ev() :
  # Gaussian distributions
  mxa = ma1, ma2 = 0.3, 0.7
  mxb = mb1, mb2 = 0.3, 0.7
  mm = moya1, moya2, moyb1, moyb2 = 0.3, 0.6, 0.7, 0.5
  ss = stda1, stda2, stdb1, stdb2 = 0.08, 0.04, 0.06, 0.04

  # melanges
  if nmb_gaussians == 1 :
    ma1, mb1, ma2, mb2 = 1, 1, 0, 0
  else :
    ma1, mb1 = np.random.uniform(low=0., high=1., size=2)
    ma2, mb2 = (1. - ma1), (1. - mb1)

  # ecart type
  ss = stda1, stda2, stdb1, stdb2 = np.random.uniform(low=0.01, high=0.1, size=4)
  
  # centre
  # on se tient à distance des bords ...
  margin = 4.5 # cette valeur permet de conserver assez souvent toute la distrib a 1e-6 pres
  mm = moya1, moya2, moyb1, moyb2 = np.random.uniform(low=ss * margin,
                                                      high=1. - ss * margin,
                                                      size=4)


  mx = ma1, ma2, mb1, mb2
  r = ev_(mx, mm, ss)
  
  (cst1, mx, mm, ss), (a, b, M, G0, cst2) = r
  return r


errs = []

def ev1(_) :
  EKO()
  while True :
    try :
      r = ev()[0]
      return r
    except Exception as ex :
      EKOX(ex)
      errs.append(1)
      pass
  
#_, (a, b, M, G0, cst) = ev()
 
print(datetime.now())


fn = lambda n : "/mnt/hd3/data/generated/ot_%04d_NG%02d.pckl" % (n, nmb_gaussians)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(  description='OT generator')
  parser.add_argument('--root', default="F")
  args = parser.parse_args()
  EKO()
  EKO()

  ev
  for _ in range(20) : ev1(1);
  sys.exit(0)
  
  def fp(n) :
    EKON(n)
    l = list(map(ev1, range(20_000)))
    print(datetime.now())
    with open(fn(n), 'wb') as fo :
      pickle.dump(l, fo)
      fo.close()
  #for n in range(10) : fp(n)
  EKO()
  num_cores = multiprocessing.cpu_count()    
  output = Parallel(n_jobs=4)(delayed(fp)(i) for i in range(10))
      
  EKOX(len(errs))
