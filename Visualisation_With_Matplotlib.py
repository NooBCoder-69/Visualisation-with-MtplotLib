#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data Visualisation with MATPLOTLIB
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[2]:


plt.style.use('classic')


# In[3]:


#Creating an example figure
import numpy as np
x=np.linspace(0,10,1000)

plt.plot(x,np.sin(x),'--')
plt.plot(x,np.cos(x))
plt.show()


# In[ ]:





# In[4]:


#Subplots
plt.figure()


# In[5]:


plt.subplot(2,1,1)
plt.plot(x,np.sin(x))
plt.title("first plot")

plt.subplot(2,1,2)
plt.plot(x,np.cos(x))
plt.title("second plot")


# In[6]:


# Trying different plots
plt.style.use('seaborn-whitegrid')


# In[7]:


fig=plt.figure()
ax=plt.axes()


# In[8]:


x=np.linspace(0,10,1000)
plt.plot(x,np.sin(x))


# In[9]:


# with  x and y limits
plt.plot(x,np.sin(x))
plt.axis([-1,11,-1.5,1.5])
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title("Sinx Graph")


# In[10]:


# A different way
plt.plot(x,np.sin(x),'-g',label='sinx')
plt.plot(x,np.cos(x),':b',label='cosx')
plt.axis('equal')
plt.legend()


# In[11]:


# Simple Scatter plots
#Scatter plot example
x=np.linspace(0,10,30)
y=np.sin(x)
plt.plot(x,y,'o',color='black')


# In[12]:


plt.plot(x, y, '-p', color='gray',
 markersize=15, linewidth=4,
 markerfacecolor='white',
 markeredgecolor='gray',
 markeredgewidth=2)
plt.ylim(-1.2, 1.2)


# In[13]:


#Scatter plot using plt.scatter
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
cmap='viridis')
plt.colorbar()


# In[14]:


#Histograms


# In[15]:


data=np.random.randn(1000)
plt.hist(data)


# In[16]:


#Different histograms comparison
x1=np.random.normal(0,0.8,1000)
x2=np.random.normal(-2,1,1000)
x3=np.random.normal(3,2,1000)
kwargs = dict(histtype='stepfilled', alpha=0.3, bins=40)
plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs)


# In[17]:


#3D plotting
fig = plt.figure()
ax = plt.axes(projection='3d')


# In[18]:


ax = plt.axes(projection='3d')
# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')
 # Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')


# In[19]:


#3D contour plots
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');


# In[ ]:


# Geographic Data with Map
get_ipython().system('conda install basemap')
from mpl_toolkits.basemap import Basemap


# In[ ]:


plt.figure(figsize=(8, 8))
m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=-100)
m.bluemarble(scale=0.5);


# In[ ]:




