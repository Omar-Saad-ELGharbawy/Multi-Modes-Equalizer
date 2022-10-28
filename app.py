import streamlit as st
import  streamlit_vertical_slider  as svs

import numpy as np
import pandas as pd

import scipy.io.wavfile as wavfile
from scipy import fftpack

import plotly.graph_objects as go

from utils import read_csv, read_wav

st.set_page_config(
    page_title="Equalizer",
    layout="wide")


if "slider1" not in st.session_state:
    st.session_state.slider1=0
if "slider2" not in st.session_state:
    st.session_state.slider2=0
if "slider3" not in st.session_state:
    st.session_state.slider3=0
if "slider4" not in st.session_state:
    st.session_state.slider4=0
if "slider5" not in st.session_state:
    st.session_state.slider5=0

if "slider6" not in st.session_state:
    st.session_state.slider6=0
if "slider7" not in st.session_state:
    st.session_state.slider7=0
if "slider8" not in st.session_state:
    st.session_state.slider8=0
if "slider9" not in st.session_state:
    st.session_state.slider9=0
if "slider10" not in st.session_state:
    st.session_state.slider10=0


# Initialization of Session State attributes (time,uploaded_signal)
if 'time' not in st.session_state:
    st.session_state.time =np.linspace(0,5,2000)
if 'signal' not in st.session_state:
    st.session_state.signal = np.sin(2*np.pi*st.session_state.time)
if 's_rate' not in st.session_state:
    st.session_state.s_rate = np.sin(2*np.pi*st.session_state.time)


file=st.file_uploader(label="Upload Signal File", key="uploaded_file",type=["csv","wav"])
## Add css design
if file:
    if file.name.split(".")[-1]=="wav":
        signal, time=read_wav(file)
        s_rate, signal1 = wavfile.read(file) 
        st.session_state.signal=signal
        st.session_state.time= time
        st.session_state.s_rate= s_rate
    elif file.name.split(".")[-1]=="csv":
        try:
            signal, time, fmax=read_csv(file)
            st.session_state.uploaded_signal=signal
            st.session_state.time= time
            st.session_state.uploaded_fmax= fmax
        except:
            st.error("Import a file with X as time and Y as amplitude")

#upload and view signal
time_signal_graph, fourier_signal_graph = st.columns([ 3, 3])


#column to draw time graph
with time_signal_graph:
    
    time=np.linspace(0,5,2000)
    full_signals=np.zeros(time.shape)
    if file:
        full_signals, time= st.session_state.signal, st.session_state.time
    else:
        time= np.linspace(0, 4, 2000)

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=time,
                                y=full_signals,
                                mode='lines',
                                name='Signal'))
    
    # fig.update_xaxes(showgrid=True, zerolinecolor='black', gridcolor='lightblue', range = (-0.1,time[-1]))
    # fig.update_yaxes(showgrid=True, zerolinecolor='black', gridcolor='lightblue', range = (-1*(max(full_signals)+0.1*max(full_signals)),(max(full_signals)+0.1*max(full_signals))))
    # fig.update_layout(
    #         font = dict(size = 20),
    #         xaxis_title="Time (sec)",
    #         yaxis_title="Amplitude",
    #         height = 600,
    #         margin=dict(l=0,r=0,b=5,t=0),
    #         legend=dict(orientation="h",
    #                     yanchor="bottom",
    #                     y=0.92,
    #                     xanchor="right",
    #                     x=0.99,
    #                     font=dict(size= 18, color = 'black'),
    #                     bgcolor="LightSteelBlue"
    #                     ),
    #         paper_bgcolor='rgb(255, 255, 255)',
    #         plot_bgcolor='rgba(255,255,255)'
    #     )
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig,use_container_width=True)



#column to draw fourier graph
with fourier_signal_graph:
    
    FFT = abs(fftpack.fft(st.session_state.signal))
    freqs = fftpack.fftfreq(len(FFT), (1.0/st.session_state.s_rate))   

    fig = go.Figure()
    # y=fftpack.fft(full_signals)
    fig.add_trace(go.Scatter(x=freqs[range(len(FFT)//2)],
                                y=FFT[range(len(FFT)//2)],
                                mode='lines',
                                name='fourier'))
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig,use_container_width=True)


# sliders columns
slider_1, slider_2, slider_3, slider_4, slider_5, slider_6, slider_7, slider_8, slider_9, slider_10 ,check_boxes = st.columns([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

with slider_1 :
    s1= svs.vertical_slider(key="slider1", default_value=50, step=1, min_value=0, 
                    max_value=100,   slider_color= 'green',#optional
                    track_color='lightgray', #optional
                    thumb_color = 'red' #optional
                    )

with slider_2 :
    s2= svs.vertical_slider(key="slider2", default_value=10, step=1, min_value=0, 
                    max_value=100,   slider_color= 'red',#optional
                    track_color='lightgray', #optional
                    thumb_color = 'green' #optional
                    )

with slider_3 :
    s3= svs.vertical_slider( key="slider3", default_value=10, step=1, min_value=0, 
                    max_value=100,   slider_color= 'red',#optional
                    track_color='green', #optional
                    thumb_color = 'blue' #optional
                    )
                    # key=key, 

with slider_4 :
    s4= svs.vertical_slider( key="slider4", default_value=10, step=1, min_value=0, 
                    max_value=100,   slider_color= 'red',#optional
                    track_color='green', #optional
                    thumb_color = 'blue' #optional
                    )


with slider_5 :
    s5= svs.vertical_slider( key="slider5", default_value=10, step=1, min_value=0, 
                    max_value=100,   slider_color= 'red',#optional
                    track_color='green', #optional
                    thumb_color = 'blue' #optional
                    )

with slider_6 :
    s1= svs.vertical_slider(key="slider6", default_value=50, step=1, min_value=0, 
                    max_value=100,   slider_color= 'green',#optional
                    track_color='lightgray', #optional
                    thumb_color = 'red' #optional
                    )

with slider_7 :
    s2= svs.vertical_slider(key="slider7", default_value=10, step=1, min_value=0, 
                    max_value=100,   slider_color= 'red',#optional
                    track_color='lightgray', #optional
                    thumb_color = 'green' #optional
                    )

with slider_8 :
    s3= svs.vertical_slider( key="slider8", default_value=10, step=1, min_value=0, 
                    max_value=100,   slider_color= 'red',#optional
                    track_color='green', #optional
                    thumb_color = 'blue' #optional
                    )
                    # key=key, 

with slider_9 :
    s4= svs.vertical_slider( key="slider9", default_value=10, step=1, min_value=0, 
                    max_value=100,   slider_color= 'red',#optional
                    track_color='green', #optional
                    thumb_color = 'blue' #optional
                    )


with slider_10 :
    s5= svs.vertical_slider( key="slider10", default_value=10, step=1, min_value=0, 
                    max_value=100,   slider_color= 'red',#optional
                    track_color='green', #optional
                    thumb_color = 'blue' #optional
                    )

with check_boxes:
    frequency = st.checkbox('frequency', value= True)  
    instruments = st.checkbox('instruments',value=False) 


# st.write(s1)
# st.write(s2)
# st.write(s3)

# st.write(st.session_state.sx)