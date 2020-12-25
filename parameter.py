#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:26:14 2020

@author: t.yamamoto
"""

SR = 16000
FFT_SIZE = 2**10
H = 2**8
PATCH_LENGTH = 128

target_path = "/disk107/DATA/CSJ_RAW/WAV/WAV"
noise_path = "/disk107/Datasets/UrbanSound8K"

MUSDB_path = "D:\yamamoto\音源分離用データ\MUSDB"
MUSDB_fft = "./Unet_MUSDB_fft/"
MUSDB_model = "./Unet_MUSDB_model/"


path_fft = "./Unet_fft/"

model_path = "./Unet_model/"

Argmentation = False
Arg_times = 2

batch_size = 20
learning_rate = 0.002
epochs = 200
