#!/usr/bin/env python3
# Author: Armit
# Create Time: 2021/09/12 

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from librosa import griffinlim
from scipy.io import wavfile

# approximately the volumn of generated wavform
SPEC_MAX_VALUE = 4
# repeat the image for better viewspect
N_REPEAT = 3
# whether convert y-axis to log scale 
#   which meaningly behaves more like spectrum in audio side
#   BUT looks ugly in viewspect in image(spetrum) side
LOG_SCALE = False
# whether show use matplotlib
DEBUG_MODE = False

# DSP Setting
SAMPLE_RATE = 44100
N_FFT = 2048
N_FREQ = N_FFT // 2 + 1
WIN_LENGTH = 1024
HOP_LENGTH = WIN_LENGTH // 4
GRIFFINLIM_ITERS = 30

# Folders
IMAGE_EXT = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

def show_specwav(S, y, log_scale=LOG_SCALE):
  plt.subplot(121)
  sns.heatmap(S, cbar=False)
  if log_scale: plt.gca().yscale("log")
  plt.gca().invert_yaxis()
  plt.subplot(122)
  plt.plot(y)
  plt.show()

def image_to_wav(fp, log_scale=LOG_SCALE, debug=DEBUG_MODE):
  # Preprocess
  with Image.open(fp) as img:
    # convert to grey
    img = img.convert(mode='L')
    # resize (w, h) to (?, N_FREQ) keeps hw-ratio
    img = img.resize((img.width * N_FREQ // img.height, N_FREQ))

  # Spectrumize
  S = np.array(img)                          # [n_freq, n_frame]
  # re-mapping y-axis to log-scale, frame sample method for interp :(
  if log_scale:
    f = lambda i: round(np.exp(np.log(N_FREQ) / N_FREQ * (i + 1)) - 1)   # mapping linear to log
    T = np.empty_like(S)
    for i in range(N_FREQ): T[i] = S[f(i)]
    S = T
  S = S[::-1, :]                             # revert in freq dim
  _S = S                                     # save a single copy for debug
  if N_REPEAT > 1: S = np.hstack([S] * N_REPEAT)
  S = (S - S.min()) / (S.max() - S.min())    # MinMax normalize to [0,1]
  S = np.log(S + 1e-5) * SPEC_MAX_VALUE      # THIS IS MAGIC !
  
  # Griffin-Lim: linear spectrum to wavform
  # NOTE: disable momentum, cos' fake spectrums DO NOT EVER converge (xswl
  y = griffinlim(S, n_iter=GRIFFINLIM_ITERS,
                 hop_length=HOP_LENGTH, win_length=WIN_LENGTH, momentum=0)
  wavfile.write(os.path.join(DATA_DIR, os.path.splitext(os.path.basename(fp))[0] + '.wav'), 
                SAMPLE_RATE, y.astype(np.float32))
  
  # debug
  if debug: show_specwav(_S, y)

if __name__ == '__main__':
  try:
    if len(sys.argv) >= 2:
      print(f'converting {sys.argv[1]} ...')
      image_to_wav(sys.argv[1])
    else:
      for fn in os.listdir(DATA_DIR):
        base, ext = os.path.splitext(fn)
        if ext.lower() not in IMAGE_EXT: continue

        if not os.path.exists(os.path.join(DATA_DIR, base + '.wav')):
          print(f'converting {fn} ...')
          image_to_wav(os.path.join(DATA_DIR, fn))
  
  except Exception as e:
    print(f'Usage: {sys.argv[0]} [image_file]')
    print(e)
