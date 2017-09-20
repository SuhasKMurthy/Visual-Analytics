from bokeh.io import output_notebook, show
from bokeh.plotting import figure
import pandas as pd
import numpy as np
from bokeh.transform import jitter
from bokeh.models import ColumnDataSource, HoverTool, Dropdown, PreText
from bokeh.palettes import inferno
from bokeh.layouts import widgetbox, gridplot, column
from bokeh.io import curdoc
from scipy import stats

doc = curdoc()

#read the dataset with missing values
dforig = pd.read_csv('Wholesale customers data-missing.csv')

#get the location of missing values
cellnull = np.where(pd.isnull(dforig))

coordinates = []
for i in range(len(cellnull[0])):
    coordinates.append((cellnull[0][i], cellnull[1][i]))

#read the original complete dataset
dfcomplete = pd.read_csv('Wholesale customers data.csv')

#create additional columns in the dataframe for the imputed values and original values of the data
df1 = dforig.copy(deep=True)

df1 = df1.assign(imputed_val=np.NaN)
df1 = df1.assign(original_val=np.NaN)

source = ColumnDataSource(df1)

heading = PreText(text="""APPLICATION OF LINEAR REGRESSION """)

menu = [("Milk vs Fresh","Milk" ), ("Grocery vs Fresh","Grocery" ), ("Detergents_Paper vs Fresh","Detergents_Paper" ), ("Delicassen vs Fresh","Delicassen")]
dropdown = Dropdown(label="Please select a Comparison", button_type="warning", menu=menu)

pempty = figure(width=1000, height=400, active_scroll="wheel_zoom")

pre = PreText(text="""A linear regression equation is derived from the column with missing data versus the data in the 'Fresh' column, in the dataset. 
The missing data is imputed from the derived equation.
Attempt is made to correlate missing data, with data in 'Fresh' column (as this attribute contains all values in the dataset). """, width=1000, height=500)

layout = column(heading, dropdown, pempty, pre)
doc.add_root(layout)

dict = {'Milk':3,'Grocery':4,'Detergents_Paper':6,'Delicassen':7}

#when user selects an item from the dropdown
def callback(attr, old, new):
    val = dropdown.value
    print(val)
    
    dfIncomplete = df1.dropna(subset=[val], inplace=False)
    x = dfIncomplete["Fresh"]
    xrange = int(df1['Fresh'].mean()) * 3
    y = dfIncomplete[val]
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    yrange = int(df1[val].mean()) * 3
    r_x, r_y = zip(*((i, i*gradient + intercept) for i in range(xrange)))
    print(gradient, intercept)
    
    
    p1 = figure(plot_width=1000, plot_height=400, title=val + " vs Fresh", x_range = (0, xrange), y_range = (0, yrange), x_axis_label='Fresh', y_axis_label=val)
    p1.line(x=r_x, y=r_y, line_width=3)

    p1.scatter(x='Fresh', y=val, source=source, marker="circle", color="blue", alpha=0.3)
    
    #add glyphs for the imputed and original data values
    for i in range(len(coordinates)):
        if coordinates[i][1] == dict.get(val):
            df1.iat[coordinates[i][0], 8] = intercept + gradient * df1.iat[coordinates[i][0], 2]
            df1.iat[coordinates[i][0], 9] = dfcomplete.iat[coordinates[i][0], coordinates[i][1]]
            print(df1.iat[coordinates[i][0], 8])
            print(df1.iat[coordinates[i][0], 9])
            imp_mean = p1.triangle(x=df1.iat[coordinates[i][0], 2], y=df1.iat[coordinates[i][0], 8],size=10,color='Black', legend='Imputed Value', alpha=1)
            p1.square(x=df1.iat[coordinates[i][0], 2], y=df1.iat[coordinates[i][0], 9],size=10,color='Red', legend='Actual Value', alpha=1)
            layout.children[2] = p1 
    
dropdown.on_change('value', callback)
