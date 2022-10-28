import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile
# import base64
# fast fourier transform
from scipy import fftpack

def read_wav(file):
    try:
        sample_rate, samples = wavfile.read(file)
        time= np.linspace(0,samples.shape[0]/sample_rate,samples.shape[0] )
        return samples, time
    except:
        time=np.linspace(0,5,2000)
        full_signals=np.zeros(time.shape)
        return full_signals, time


def read_csv(file):
    try:
        df= pd.read_csv(file)
        signal= np.array(df['Y'])
        time= np.array(df["X"])
        return signal,time
        # return signal,time
    except:
        return ValueError("Import a file with X as time and Y as amplitude")

#fftpack.fft(arr)