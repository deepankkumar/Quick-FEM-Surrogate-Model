#!/usr/bin/env python
# coding: utf-8

# In[14]:
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import pickle as cPickle   
import matplotlib as mpl
from mpl_toolkits import mplot3d
import plotly.express as px
mpl.use("agg")
# -- Set page config
apptitle = 'QFSM'
st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:",layout='wide')
# -- Default detector list

detectorlist = ['H1','L1', 'V1']
@st.cache(ttl=3600, max_entries=10)   #-- Magic command to cache data
def load_gw(t0, detector):
    strain = TimeSeries.fetch_open_data(detector, t0-14, t0+14, cache=False)
    return strain

# Title the app
st.title('Quick FEM Surrogate Model')
expander = st.beta_expander("About")
expander.write("In this project we have developed the ML-model for the prediction of Stress and deformation on a 3-D cantilever beam. This will accelerate the product development workflow by dramatically reducing the speed of FEM analysis.")
st.header('Final Year Project')
st.markdown("""
 * Use the menu at left to select data and set plot parameters
 * Your plots will appear below""")
with st.sidebar.beta_expander("Contributors"):

    st.markdown("""
 * [Dr. Ravindra Nagar | PI]()
 * [Deepank Singh]()
 * [Udbhav Tripathi]()
 * [Abhijeet Yadav]()
""")

st.sidebar.markdown("## Select the input data")

Dim = st.sidebar.radio("What type of Result Visualization you want? : ",('3D - Slower', '2D - Faster'))
Res = st.sidebar.radio("What result do you need ? ",('Stress', 'Deformation'))
st.sidebar.subheader("Select the Combination:")
st.sidebar.subheader("Select the Combination:")
height=st.sidebar.slider('Set the height', 200, 500, 300,10)
load_position=st.sidebar.slider('Set the load position', 200, 1000, 300,10)
load_value=st.sidebar.slider('Set the load Value', 100, 1000, 300,10)


if Dim == '2D - Faster':
    # Defining the dimensions of a beam, load values and load position.
    breadth=200
    # Creating the 2D- beam
    X=[[100]]*int((height+10)*0.1)*101
    Y=[[y] for y in range(0,height+10,10)]*101    
    Z=[[z]*int((height+10)*0.1) for z in range(0,1010,10)]
    # Conversion of list data into numpy array form
    X=np.array(X)
    Y=np.array(Y)
    Z=np.array(Z)
    # Using ravel, for flattening of multi dimensional array 
    X=np.ravel(X)
    Y=np.ravel(Y)
    Z=np.ravel(Z)
    df=pd.DataFrame({"X Location (mm)":X,"Y Location (mm)":Y,"Z Location (mm)":Z,"B":breadth,"H":height,"X":load_position,"P":load_value})
    if Res == 'Stress':
        
        with open('Stress_nn.pkl', 'rb') as fid:
            reg = cPickle.load(fid)
        with open('stndrd_scaler.pkl', 'rb') as fid:
            standard = cPickle.load(fid)
        with open('Stress_colm.pkl', 'rb') as fid:
            colms = cPickle.load(fid)
        df=standard.transform(df)
        df1 = pd.DataFrame(data=df, columns=colms)
        df1['Equivalent (von-Mises) Stress (MPa)']=reg.predict(df1)
        df2=standard.inverse_transform(df1[df1.columns[:-1]])
        df2 = pd.DataFrame(data=df2, columns=colms)
        df2['Equivalent (von-Mises) Stress (MPa)']=df1['Equivalent (von-Mises) Stress (MPa)']
        df2.drop('X Location (mm)',axis=1,inplace=True)
        import matplotlib as mpl
        import matplotlib
        matplotlib.use('Agg')
        zlist=np.linspace(0,1000,101)
        ylist=np.linspace(0,height,int((height+10)*0.1))
        Y, Z = np.meshgrid(ylist, zlist)
        C=np.reshape(df2['Equivalent (von-Mises) Stress (MPa)'].values,(101,int((height+10)*0.1)))      # For 2D problem, keep check on 
        cmap = mpl.colors.ListedColormap(['red', 'green', 'blue', 'cyan'])
        plt.figure()
        plt. clf()
        fig=plt.figure()
        ax=fig.add_axes([0,10,1.5,1])
        ax.annotate('LOAD={} kN'.format(load_value),xy=(load_position,height),xytext=(load_position-100,height+50)
                    ,arrowprops=dict(facecolor='red',shrink=0.01*load_value),fontsize=15)
        ax.annotate('FIXED END',xy=(0,0),xytext=(50,0),
                    arrowprops=dict(facecolor='red',shrink=0.01*load_value),fontsize=20)    
        plt.axis('equal')
        # ax.set_xticks([0,100,200,300,400,500,600,700,800,900,1000])
        points = plt.scatter(Z,Y,c=C,cmap='rainbow',lw=0)
        plt.colorbar(points)
        from matplotlib import pyplot as plt
        plt.show()
        max = df2['Equivalent (von-Mises) Stress (MPa)'].max()
        st.subheader("2D: Stress Plot | Max Absolute Stress: "+str(max)+"Mpa")
        st.text(' ')
        st.pyplot(fig=fig, clear_figure=None)
    elif Res == 'Deformation':
        
        with open('defor_nn.pkl', 'rb') as fid:
            reg = cPickle.load(fid)
        with open('stndrd_scaler_defor.pkl', 'rb') as fid:
            standard = cPickle.load(fid)
        with open('Stress_colm_defor.pkl', 'rb') as fid:
            colms = cPickle.load(fid)
        df=standard.transform(df)
        df1 = pd.DataFrame(data=df, columns=colms)
        df1['Total Deformation (mm)']=reg.predict(df1)
        df2=standard.inverse_transform(df1[df1.columns[:-1]])
        df2 = pd.DataFrame(data=df2, columns=colms)
        df2['Total Deformation (mm)']=df1['Total Deformation (mm)']
        df2.drop('X Location (mm)',axis=1,inplace=True)
        
        zlist=np.linspace(0,1000,101)
        ylist=np.linspace(0,height,int((height+10)*0.1))
        Y, Z = np.meshgrid(ylist, zlist)
        C=np.reshape(df2['Total Deformation (mm)'].values,(101,int((height+10)*0.1)))      # For 2D problem, keep check on 
        cmap = mpl.colors.ListedColormap(['red', 'green', 'blue', 'cyan'])
        plt.figure()
        fig=plt.figure()
        ax=fig.add_axes([0,10,1.5,1])
        ax.annotate('LOAD={} kN'.format(load_value),xy=(load_position,height),xytext=(load_position-100,height+50)
                    ,arrowprops=dict(facecolor='red',shrink=0.01*load_value),fontsize=15)
        ax.annotate('FIXED END',xy=(0,0),xytext=(50,0),
                    arrowprops=dict(facecolor='red',shrink=0.01*load_value),fontsize=15)    
        plt.axis('equal')
        # ax.set_xticks([0,100,200,300,400,500,600,700,800,900,1000])
        points = plt.scatter(Z,Y-C*1000,c=C,cmap='rainbow',lw=0)
        plt.colorbar(points)
        from matplotlib import pyplot as plt
        plt.show()
        max = df2['Total Deformation (mm)'].max()
               
        st.subheader("2D: Deformation Plot | Max Deformation: "+str(max)+"mm")
        st.text(' ')
        st.pyplot(fig=fig, clear_figure=None)
        
if Dim == '3D - Slower':       
    # Defining the dimensions of a beam, load values and load position.
    breadth=200
    # Creating the 3D- beam
    X=[[x] for x in range(0,210,10)]*int((height+10)*0.1)*101
    Y=[[y]*21 for y in range(0,height+10,10)]*101    
    Z=[[z]*21*int((height+10)*0.1) for z in range(0,1010,10)]
    # Conversion of list data into numpy array form
    X=np.array(X)
    Y=np.array(Y)
    Z=np.array(Z)
    # Using ravel, for flattening of multi dimensional array 
    X=np.ravel(X)
    Y=np.ravel(Y)
    Z=np.ravel(Z)
    # Generating the input DataFrame for model
    df=pd.DataFrame({"X Location (mm)":X,"Y Location (mm)":Y,"Z Location (mm)":Z,"B":breadth,"H":height,"X":load_position,"P":load_value})
        # Add a placeholder
    if Res == 'Stress':
        with open('Stress_nn.pkl', 'rb') as fid:
            reg = cPickle.load(fid)
        with open('stndrd_scaler.pkl', 'rb') as fid:
            standard = cPickle.load(fid)
        with open('Stress_colm.pkl', 'rb') as fid:
            colms = cPickle.load(fid)
        df=standard.transform(df)
        df1 = pd.DataFrame(data=df, columns=colms)
        df1['Equivalent (von-Mises) Stress (MPa)']=reg.predict(df1)
        df2=standard.inverse_transform(df1[df1.columns[:-1]])
        df2 = pd.DataFrame(data=df2, columns=colms)
        df2['Equivalent (von-Mises) Stress (MPa)']=df1['Equivalent (von-Mises) Stress (MPa)']
        fig = px.scatter_3d(df2, x='Z Location (mm)', y='Y Location (mm)', z='X Location (mm)',
                      color='Equivalent (von-Mises) Stress (MPa)',color_continuous_scale='rainbow')
        fig.update_layout(scene_aspectmode='data')
        max = df2['Equivalent (von-Mises) Stress (MPa)'].max()
        st.subheader("3D: Stress Plot | Max Absolute Stress: "+str(max)+"Mpa")
        st.text(' ')
        st.write(fig)
        
    elif Res == 'Deformation':

            with open('defor_nn.pkl', 'rb') as fid:
                reg = cPickle.load(fid)
            with open('stndrd_scaler_defor.pkl', 'rb') as fid:
                standard = cPickle.load(fid)
            with open('Stress_colm_defor.pkl', 'rb') as fid:
                colms = cPickle.load(fid)
            df=standard.transform(df)
            df1 = pd.DataFrame(data=df, columns=colms)
            df1['Total Deformation (mm)']=reg.predict(df1)
            df2=standard.inverse_transform(df1[df1.columns[:-1]])
            df2 = pd.DataFrame(data=df2, columns=colms)
            df2['Total Deformation (mm)']=df1['Total Deformation (mm)']
            fig = px.scatter_3d(df2, x='Z Location (mm)', y='Y Location (mm)', z='X Location (mm)',
                      color='Total Deformation (mm)',color_continuous_scale='rainbow')
            fig.update_layout(scene_aspectmode='data')
            max = df2['Total Deformation (mm)'].max()
            st.subheader("3D: Deformation Plot | Max Deformation: "+str(max)+"mm")
            st.text(' ')
            st.write(fig)
    
