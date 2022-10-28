import streamlit as st
import  streamlit_vertical_slider  as svs
import pandas as pd
# 
st.set_page_config(layout="wide")
st.subheader("Vertical Slider")

# if "sx" not in st.session_state:
#     st.session_state.sx=70

ce, c1, c2, c3, ce = st.columns([0.07, 3,  3, 3, 0.07])

with c1 :
    x= svs.vertical_slider(key="sx", default_value=50, step=1, min_value=0, 
                    max_value=100,   slider_color= 'green',#optional
                    track_color='lightgray', #optional
                    thumb_color = 'red' #optional
                    )

with c2 :
    y= svs.vertical_slider(default_value=10, step=1, min_value=0, 
                    max_value=100,   slider_color= 'red',#optional
                    track_color='lightgray', #optional
                    thumb_color = 'green' #optional
                    )

with c3 :
    z= svs.vertical_slider(default_value=10, step=1, min_value=0, 
                    max_value=100,   slider_color= 'red',#optional
                    track_color='green', #optional
                    thumb_color = 'blue' #optional
                    )
                    # key=key, 

# y= svs.vertical_slider(default_value=1, step=1, min_value=0, 
#                     max_value=100,   slider_color= 'green',#optional
#                     track_color='lightgray', #optional
#                     thumb_color = 'red' #optional
#                     )
                    

st.write(x)
st.write(y)
st.write(z)

# fig = go.Figure()
# st.write(st.session_state.sx)