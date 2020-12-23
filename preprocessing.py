#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:34:38 2020

@author: t.yamamoto
"""

import myutils as ut
import parameter as C
import os

def main():
    
    if not os.path.exists(C.path_fft):
        os.mkdir(C.path_fft)
        
    ut.SaveSTFT()
    
    if C.Argmentation:
        for i in range(C.Arg_times):
            pitch_shift = int((-1)**i * (5+i))
            time_stretch = int(1 + (i/3)**(-1)**i)
            ut.SaveSTFT_Arg(pitch_shift,time_stretch,i)
        
if __name__ == "__main__":
    main()