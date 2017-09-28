#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:42:35 2017

@author: suhas
"""

from bokeh.io import output_notebook, show
from bokeh.plotting import figure
import pandas as pd
import numpy as np
import math
from bokeh.transform import jitter
from bokeh.models import ColumnDataSource, HoverTool, Dropdown, PreText, Slider, Button, Label, Select
from bokeh.palettes import inferno
from bokeh.layouts import widgetbox, gridplot, column, layout, row
from bokeh.io import curdoc
from bokeh.events import ButtonClick
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

doc = curdoc()

#read the original complete dataset
dfcomplete = pd.read_csv('Wholesale customers data.csv')

#initial global values
xval = "Fresh"
yval = "Milk"
num_clus_kmeans = 1
num_minpts = 3
num_eps = 1
n_neigh = 5
num_clus_ward = 1
sel_channel = "All"
sel_region = "All"

heading = PreText(text="""CLUSTERING ALGORITHM ON WHOLESALE CUSTOMERS DATA""", height=25, width=500)

menu = [("Fresh","Fresh"), ("Milk", "Milk"), ("Grocery","Grocery" ), ("Frozen", "Frozen"), ("Detergents_Paper","Detergents_Paper" ), ("Delicassen","Delicassen")]
ddx = Dropdown(label="Select x-axis", button_type="primary", menu=menu, height=25, width=110)
ddy = Dropdown(label="Select y-axis", button_type="primary", menu=menu, height=25, width=110)

menu = [("K-Means", "0"), ("DBSCAN", "1"), ("Hierarchical", "2")]

pempty1 = figure(width=800, height=300, active_scroll="wheel_zoom")
pempty2 = figure(width=800, height=300, active_scroll="wheel_zoom")

select_channel = Select(title="Channel:", value="All", options=["All", "1", "2"], height=25, width=110)
select_region = Select(title="Region:", value="All", options=["All", "1", "2", "3"], height=25, width=110)

#widget_box_axes = widgetbox(ddx,ddy)
widget_box_axes = row(widgetbox(ddx), widgetbox(ddy), widgetbox(select_channel), widgetbox(select_region))  

empty1 = PreText(text="""\n\nK-Means """, height=50, width=200)

slider_cluster = Slider(start=1, end=10, value=1, step=1, title="No. of clusters", width=200)
button_k = Button(label="Evaluate", width=100, button_type="success")
widget_box_k_means = widgetbox(empty1, slider_cluster, button_k)

empty2 = PreText(text="""\nWard Agglomerative \nClustering""", height=50, width=200)

slider_n_neigh = Slider(start=5, end=100, value=5, step=5, title="N-neighbours", width=200)
slider_cluster_ward = Slider(start=2, end=10, value=2, step=1, title="No. of clusters", width=200)
button_ward = Button(label="Evaluate", width=100, button_type="success")
widget_box_ward = widgetbox(empty2, slider_n_neigh, slider_cluster_ward, button_ward)

#l = layout([heading],[[[ddx,ddy],[ddclus]],pempty])
l = layout([heading],[widget_box_axes],[widget_box_k_means,pempty1], [widget_box_ward,pempty2])

doc.add_root(l)

def cb_bt_kmeans():
    df1 = get_data()
    
    #print("Button event")
    f1 = df1[xval].values
    f2 = df1[yval].values
    
    colors = inferno(num_clus_kmeans)
    
    label_colors = {}
    for p in range(0,num_clus_kmeans) :
        label_colors[p] = colors[p]
    
    X = np.matrix(list(zip(f1,f2)))
    
    k_means = KMeans(n_clusters=num_clus_kmeans).fit(X)
    labels = list(k_means.labels_)
    
    df1['clusters'] = labels
    
    df1.clusters.replace(to_replace=label_colors, inplace=True)
    
    cent = k_means.cluster_centers_
    
    #print(cent)
    
    a = [row[0] for row in cent]
    b = [i[1] for i in cent]
    
    print(a, b)
    
    df1['centroid_x'] = pd.Series(a)
    df1['centroid_y'] = pd.Series(b)
    
    print(df1)
    
    src1 = ColumnDataSource(df1)
    
    p1 = figure(plot_width=800, plot_height=400, title=xval + " vs " + yval, x_axis_label=xval, y_axis_label=yval)
    p1.circle(x=xval, y=yval,size=7,color='clusters', source=src1, alpha=0.2)
    
    p1.circle_x(x='centroid_x', y='centroid_y', color='Red', legend='k-means centroid', size=12, source=src1 )
    
    #paramVal = Label(x=70, y=(max_y - (0.1 * max_y)), text='Inertia - ' + str(k_means.inertia_))

    #p1.add_layout(paramVal)
    
    #p1.legend.orientation = "horizontal"
    #p1.legend.click_policy = "hide"

    #print(l.children)
    l.children[2].children[1] = p1
    
def cb_bt_ward():
    
    df1 = get_data()
    
    print("Button event")
    f1 = df1[xval].values
    f2 = df1[yval].values
    
    X = np.matrix(list(zip(f1,f2)))
    
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=n_neigh, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    
    ward = AgglomerativeClustering(
                n_clusters=num_clus_ward, linkage='ward',
                connectivity=connectivity).fit(X)
    
    labels = list(ward.labels_)
    
    df1['clusters'] = labels
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    colors = inferno(n_clusters)
    
    
    label_colors = {}
    for p in range(0,n_clusters) :
        label_colors[p] = colors[p]
    
    df1.clusters.replace(to_replace=label_colors, inplace=True)
    
    src1 = ColumnDataSource(df1)
    
    p1 = figure(plot_width=800, plot_height=400, title=xval + " vs " + yval, x_axis_label=xval, y_axis_label=yval)
    p1.circle(x=xval, y=yval,size=7,color='clusters', source=src1, alpha=0.4)
    
    
    l.children[3].children[1] = p1

#not implemented
def cb_bt_dbscan():
    
    
    print("Button event")
    f1 = dfcomplete[xval].values
    f2 = dfcomplete[yval].values
    
    print(f1, f2)
    X = np.matrix(list(zip(f1,f2)))
    print(num_clus_kmeans)
    
    db = DBSCAN(eps=num_eps, min_samples=num_minpts).fit(X)
    labels = list(db.labels_)
    
    df1['clusters'] = labels
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    colors = inferno(n_clusters)
    
    print(colors)
    
    label_colors = {}
    for p in range(0,n_clusters) :
        label_colors[p] = colors[p]
    
    print(label_colors)
    
    df1.clusters.replace(to_replace=label_colors, inplace=True)
    
    src1 = ColumnDataSource(df1)
    
    p1 = figure(plot_width=800, plot_height=400, title=xval + " vs " + yval, x_axis_label=xval, y_axis_label=yval)
    p1.circle(x=xval, y=yval,size=7,color='clusters', legend='clusters', source=src1, alpha=0.2)
    
    l.children[3].children[1] = p1

def get_data():
    df1 = dfcomplete.copy(deep=True)
    
    if(sel_channel != "All" and sel_region != "All"):
        df1 = df1[df1['Channel'] == int(sel_channel)]
        df1 = df1[df1['Region'] == int(sel_region)]
    elif(sel_channel != "All"):
        df1 = df1[df1['Channel'] == int(sel_channel)]
    elif(sel_region != "All"):
        df1 = df1[df1['Region'] == int(sel_region)]
    
    df1.reset_index(drop=True, inplace=True)
    
    print(df1.shape)    
    return df1
    
#when user selects an item from the dropdown
def cby(attr, old, new):
    global yval
    yval = ddy.value
    plot()
    print(yval)
    
#when user selects an item from the dropdown
def cbx(attr, old, new):
    global xval
    xval = ddx.value
    plot()
    print(xval)
    
def cb_sl_clus(attr, old, new):
    global num_clus_kmeans
    num_clus_kmeans = math.ceil(slider_cluster.value)
    print(num_clus_kmeans)
    
def cb_sl_minpts(attr, old, new):
    global num_minpts
    num_minpts = math.ceil(slider_cluster.value)
    print(num_minpts)
    
def cb_sl_eps(attr, old, new):
    global num_eps
    num_eps = math.ceil(slider_cluster.value)
    print(num_eps)
    
def cb_sl_nneigh(attr, old, new):
    global n_neigh
    n_neigh = math.ceil(slider_n_neigh.value)
    print(n_neigh)
    
def cb_sl_clusward(attr, old, new):
    global num_clus_ward
    num_clus_ward = math.ceil(slider_cluster_ward.value)
    print(num_clus_ward)

def cb_sel_channel(attr, old, new):
    global sel_channel
    sel_channel = select_channel.value
    print(sel_channel)
    
def cb_sel_region(attr, old, new):
    global sel_region
    sel_region = select_region.value
    print(sel_region)
    
def plot():
    df1 = get_data()
        
    source = ColumnDataSource(df1)
    
    print("Plotting", xval, yval)
    p1 = figure(plot_width=800, plot_height=400, title=xval + " vs " + yval, x_axis_label=xval, y_axis_label=yval)
    p1.circle(x=xval, y=yval,size=7,color='Green', source=source, alpha=0.4)
    print(l.children)
    l.children[2].children[1] = p1
    
    p2 = figure(plot_width=800, plot_height=400, title=xval + " vs " + yval, x_axis_label=xval, y_axis_label=yval)
    p2.circle(x=xval, y=yval,size=7,color='Green', source=source, alpha=0.4)
    l.children[3].children[1] = p2

#on change events for the widgets
ddx.on_change('value', cbx)
ddy.on_change('value', cby)

slider_cluster.on_change('value', cb_sl_clus)
button_k.on_click(cb_bt_kmeans)

slider_n_neigh.on_change('value', cb_sl_nneigh)
slider_cluster_ward.on_change('value', cb_sl_clusward)
button_ward.on_click(cb_bt_ward)

select_channel.on_change('value', cb_sel_channel)
select_region.on_change('value', cb_sel_region)

plot()
#print(l.children[1].children[1])