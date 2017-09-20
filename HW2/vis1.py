from bokeh.io import output_notebook, show
from bokeh.plotting import figure
import pandas as pd
import numpy as np
from bokeh.transform import jitter
from bokeh.models import ColumnDataSource, HoverTool, Dropdown, PreText
from bokeh.palettes import inferno
from bokeh.layouts import widgetbox, gridplot, column
from bokeh.io import curdoc

#read the dataset with missing values
dforig = pd.read_csv('Wholesale customers data-missing.csv')

#get the location of missing values
cellnull = np.where(pd.isnull(dforig))

coordinates = []
for i in range(len(cellnull[0])):
    coordinates.append((cellnull[0][i], cellnull[1][i]))

#read the original complete dataset
dfcomplete = pd.read_csv('Wholesale customers data.csv')

d = {}

#get the mean of all the attribute columns in the dataset
for i in set(cellnull[1]):
    d[i] = int(dforig.iloc[:, i].mean(axis=0))

df1 = dforig.copy(deep=True)

#create additional columns in the dataframe for the imputed values and original values of the data
df1 = df1.assign(imputed_val=np.NaN)
df1 = df1.assign(original_val=np.NaN)

#fill the additional columns in the dataframe for the imputed values and original values of the data
for i in range(len(coordinates)):
    df1.iat[coordinates[i][0], 8] = d.get(coordinates[i][1])
    df1.iat[coordinates[i][0], 9] = dfcomplete.iat[coordinates[i][0], coordinates[i][1]]
    
df1 = df1.sort_values(['Channel','Region'], ascending=[True,True])
df1["id"] = df1["Channel"].map(str) + "," + df1["Region"].map(str)

IDS = list(df1.id.unique())

source = ColumnDataSource(df1)

#get a color list
col = inferno(6)

p = figure(plot_width=1000, plot_height=500, x_range=IDS, title="Imputed Values of Missing Data", x_axis_label='Channel,Region', y_axis_label='Annual Spending (m.u.)')
    
p.circle(x=jitter('id', width=0.6, range=p.x_range), y='Fresh', color=col[0], legend='fresh', source=source, alpha=0.2)
p.circle(x=jitter('id', width=0.6, range=p.x_range), y='Milk', color=col[1], legend='milk', source=source, alpha=0.2)
p.circle(x=jitter('id', width=0.6, range=p.x_range), y='Grocery', color=col[2], legend='grocery', source=source, alpha=0.2)
p.circle(x=jitter('id', width=0.6, range=p.x_range), y='Frozen', color=col[3], legend='frozen', source=source, alpha=0.2)
p.circle(x=jitter('id', width=0.6, range=p.x_range), y='Detergents_Paper', color=col[4], legend='detergents_paper', source=source, alpha=0.2)
p.circle(x=jitter('id', width=0.6, range=p.x_range), y='Delicassen', color=col[5], legend='delicassen', source=source, alpha=0.2)

#add glyphs for the imputed and original data values
imp_mean = p.triangle(x=jitter('id', width=0.6, range=p.x_range), y='imputed_val',size=7,color='Black', legend='Imputed Val', source=source, alpha=1)
org_val = p.square(x=jitter('id', width=0.6, range=p.x_range), y='original_val',size=7, color='Green', legend='Org Val', source=source, alpha=1)

p.legend.orientation = "horizontal"
p.legend.click_policy = "hide"

p.add_tools(HoverTool(renderers=[imp_mean], tooltips=[
        ('Imputed Mean Value',   '@imputed_val' ),
        ('Original Value', '@original_val')
    ]))

heading = PreText(text="""MEAN OF ATTRIBUTE""")

pre = PreText(text="""The missing data is imputed from calculating the mean value of the attribute column in the dataset.
Hover over the triangle to display the actual and imputed values of the data point. 
Click on the legend items to hide corresponding data points on the plot""", width=800, height=200)

curdoc().add_root(column(heading, p, pre))
