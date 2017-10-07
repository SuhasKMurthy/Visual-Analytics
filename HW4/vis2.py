#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 13:05:36 2017

@author: suhas
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:42:35 2017

@author: suhas
"""

from bokeh.plotting import figure
import pandas as pd
import numpy as np
from bokeh.transform import jitter
from bokeh.models import ColumnDataSource, HoverTool, Dropdown, PreText, Slider, Button, Label, Select, CheckboxGroup, RadioGroup
from bokeh.palettes import inferno
from bokeh.layouts import widgetbox, gridplot, column, layout, row
from bokeh.io import curdoc
from bokeh.events import ButtonClick
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import metrics

doc = curdoc()

#read the original complete dataset
dfcomplete = pd.read_csv('test.csv')

#initial global values
xval = "GROUP_A_SUGBEV_TOTAL_KCAL"
yval = "GROUP_SWEETS_TOTAL_KCAL"

txt_smoke = PreText(text="Smoking Information")

lst_smoke = ["all", "ever_smoked", "currently_smoke", "smoke_often", "smoke_rarely", "never_smoked"]
rdo_grp_smoke = RadioGroup(
        labels=lst_smoke, active=0)

lst_misc = ["all", "rash", "cat", "dog", "Dems", "atheist", "Jewish" ]
txt_misc = PreText(text="Select other criteria")

rdo_grp_misc = RadioGroup(
        labels=lst_misc, active=0)

widget_box_selection = widgetbox(txt_smoke, rdo_grp_smoke, 
                              txt_misc, rdo_grp_misc)

heading = PreText(text="""NUTRITION DATA""", height=25, width=500)

all_columns = list(dfcomplete)

z = tuple((item,item) for item in all_columns[30:])

txt_x = PreText(text=xval)

ddx = Dropdown(label="Select x-axis", button_type="primary", menu=list(z), height=25, width=200)
ddy = Dropdown(label="Select y-axis", button_type="primary", menu=list(z), height=25, width=200)

txt_y = PreText(text=yval)

button_plot = Button(label="Plot", width=100, button_type="warning")

#empty plot placeholders
pempty1 = figure(width=800, height=500, active_scroll="wheel_zoom")

#build layout
l = layout([heading], [ddx, txt_x, ddy, txt_y, button_plot], [widget_box_selection, pempty1])

doc.add_root(l)    
    
#from analytics vidhya site
#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5, random_state = 10)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome])

#outcome_var = yval
##
#model = RandomForestClassifier(n_estimators=500, oob_score=True, max_features=0.001)
###model = LogisticRegression()
###model = PCA(n_components=2, svd_solver='full')
#predictor_var = list(dfcomplete)[30:]
###
#classification_model(model, dfcomplete, predictor_var, outcome_var)	
#featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
#
##print(featimp)
#print(list(featimp.index.values)[:15])
#
##y_pred = gnb.fit(predictor_var, dfcomplete[outcome_var]).predict(predictor_var)
##print(y_pred)
#
#model = RandomForestClassifier(n_estimators=200, oob_score=True, max_features=0.01)
#predictor_var = list(featimp.index.values)[:10]
#classification_model(model, dfcomplete, predictor_var,outcome_var)
#featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
#print(list(featimp.index.values)[:5])
#print(featimp)
#

def get_data():
    df1 = dfcomplete.copy(deep=True)
    

    mapping_can = {'No': 'Black', 'Yes': 'Red'}
    mapping_dia = {'No': 'Black', 'Yes': 'Blue'}
    mapping_hea = {'No': 'Black', 'Yes': 'Yellow'}

    df1 = df1.replace({'cancer': mapping_can, 'diabetes': mapping_dia, 'heart_disease':mapping_hea})
    
    sel_smoke = lst_smoke[rdo_grp_smoke.active]
    
    if(sel_smoke != 'all'):
        df2 = df1[df1[lst_smoke[rdo_grp_smoke.active]] == 'Yes']
    else:
        df2 = df1.copy()
        
    sel_misc = lst_smoke[rdo_grp_misc.active]
    
    if(sel_misc != 'all'):
        df3 = df2[df2[lst_misc[rdo_grp_misc.active]] == 'Yes']
    else:
        df3 = df2.copy()
    
    return df3

def plot():
    df1 = get_data()
        
    source = ColumnDataSource(df1)
    
    p1 = figure(plot_width=800, plot_height=500, title=xval + " vs " + yval, x_axis_label=xval, y_axis_label=yval)
    p1.annulus(x=xval, y=yval,color='cancer', inner_radius=0.1, outer_radius=2, line_color="white", legend='Cancer', source=source, alpha=0.6)
    p1.annulus(x=xval, y=yval,color='diabetes', inner_radius=2, outer_radius=4,
               line_color="white", source=source, legend='Diabetes', alpha=0.6)
    p1.annulus(x=xval, y=yval,color='heart_disease', inner_radius=4, outer_radius=6,
               line_color="white", legend='Heart Disease', source=source, alpha=0.6)
    
    p1.legend.orientation = "horizontal"
    p1.legend.click_policy = "hide"
    
    f1 = df1[xval].values
    f2 = df1[yval].values
    
    colors = inferno(3)
    
    label_colors = {}
    for p in range(0,3) :
        label_colors[p] = colors[p]
    
    X = np.matrix(list(zip(f1,f2)))
    
    k_means = KMeans(n_clusters=3).fit(X)
    labels = list(k_means.labels_)
    
    df1['clusters'] = labels
    
    df1.clusters.replace(to_replace=label_colors, inplace=True)
    
    #get centroids and add to dataframe
    cent = k_means.cluster_centers_
    
    a = [row[0] for row in cent]
    b = [i[1] for i in cent]
    
    df1['centroid_x'] = pd.Series(a)
    df1['centroid_y'] = pd.Series(b)
    
    src1 = ColumnDataSource(df1)
    
    #p1 = figure(plot_width=800, plot_height=400, title=xval + " vs " + yval, x_axis_label=xval, y_axis_label=yval)
    p1.x(x=xval, y=yval,size=7,color='clusters', source=src1, alpha=1)
    
    p1.triangle(x='centroid_x', y='centroid_y', color='Red', legend='k-means centroid', size=12, source=src1 )
    
    l.children[2].children[1] = p1

plot()
    
#when user selects an item from the dropdown
def cby(attr, old, new):
    global yval
    yval = ddy.value
    txt_y.text = yval
    #plot()
    print(yval)
    
#when user selects an item from the dropdown
def cbx(attr, old, new):
    global xval
    xval = ddx.value
    txt_x.text = xval
    #plot()
    print(xval)
    

def cb_bt_plot():
    plot()
    
def cb_rdo_smoke(new):
    print(rdo_grp_smoke.active)

def cb_rdo_misc(new):
    print(rdo_grp_misc.active)
    
#on change events for the widgets
ddx.on_change('value', cbx)
ddy.on_change('value', cby)

rdo_grp_smoke.on_click(cb_rdo_smoke)
rdo_grp_misc.on_click(cb_rdo_misc)

button_plot.on_click(cb_bt_plot)