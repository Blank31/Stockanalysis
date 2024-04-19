# %%
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly
import yfinance as yfi
from dash import Dash, html, callback, Output, Input, dcc
import dash as dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import mpld3


from statsmodels.tsa.arima.model import ARIMA
from statsmodels import tsa
from pmdarima.arima import auto_arima
from sklearn.linear_model import ElasticNet
import datetime

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css"
app = dash.Dash(external_stylesheets=[dbc.themes.CYBORG])


### -------------------------------------------------------- ###
### -----------------EXTRA STUFF - LUKE -------------------- ###
### -------------------------------------------------------- ###
stock_list = ['AAPL','AMZN','NVDA','MSFT','IBM','INTC']

slider_date_yfi_dict = {0:'1d' ,1:'1wk',2:'1mo',
                    3:'6mo',4:'1y',5:'5y'}
slider_date_yfi_interval_dict = {0:['5m','1 Day'],1:['30m','1 Week'],
                                 2:['30m','1 Month'],3:['1h','6 Months'],
                                 4:['1d','1 Year'],5:['1d','5 Years']}

radio_yfi_list = ['Open Price','Volume of Shares Sold','Gross Profit'] ### THIS IS MY BUTTON NAMES
radio_yfi_data_dict = {'Open Price':'Open', 'Volume of Shares Sold':'Volume'}

### -------------------------------------------------------- ###
### -----------------EXTRA STUFF - DURGESH ----------------- ###
### -------------------------------------------------------- ###

radio_yfi_list_tab2 = ['None','ElasticNet','ARIMA','Both'] ### THIS IS MY BUTTON NAMES


### -------------------------------------------------------- ###
### -----------------EXTRA STUFF - MICHAEL------------------ ###
### -------------------------------------------------------- ###

radio_yfi_list_tab3 = ['Open Price','Volume of Shares Sold'] ### THIS IS MY BUTTON NAMES


app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label="Tab 1 - Stock Data Visualizations", children = [
            html.Div([
            html.H1(children='Stock data visualization', style={'textAlign':'center'}),
            dcc.Dropdown(stock_list, 'AMZN', id='DROPDOWN-SELECTION', clearable=False)]),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id='GRAPH-CONTENT',responsive=False)), width=9),  
                dbc.Col(html.Div(dcc.RadioItems(options = radio_yfi_list, value = 'Open Price',id = 'RADIO-SELECTION',
                                        style={'textAlign':'center', 'font-size':40}),
                                style={'border':'2px white solid', 'height':'100%'}), width=3)       
                    ]),
            dcc.Slider(0,5,step=None,marks=slider_date_yfi_dict, value=0,id='SLIDER-SELECTION')
                    ]),

                   # ---------------------------------------------- #
                   ### Durgesh code here for structuring the page!!!
                   # ---------------------------------------------- #
        dcc.Tab(label='Tab 2 - Stock price forecasting',
               children = [
                html.Div([
                html.H1(children='Stock Data Forecasting', style={'textAlign':'center'}),
                dcc.Dropdown(stock_list, 'AMZN', id='DROPDOWN-SELECTION-2', clearable=False)]),
                dbc.Row([
                    dbc.Col(html.Div(dcc.Graph(id='GRAPH-CONTENT-2',responsive=False)), width=9),  
                    dbc.Col(html.Div([dcc.RadioItems(options = radio_yfi_list_tab2, value = 'None',id = 'RADIO-SELECTION-2',
                                        style={'textAlign':'center', 'font-size':40}),
                                      dcc.Input(value=1, type='number',id = 'K-SELECTION-2')]),
                                style={'border':'2px white solid', 'height':'100%'},width=3)]) 
               ]),

                    # --------------------------------------------- #
                   ### Michael code here for structuring the page!!!
                   # ---------------------------------------------- #
        dcc.Tab(label='Tab 3 - Stock price decomposition', children=[
            html.Div([
                html.H1(children='Stock Data Decomposition', style={'textAlign':'center'}),
                dcc.Dropdown(stock_list, 'AMZN', id='DROPDOWN-SELECTION-3', clearable=False)]),
                dbc.Row([
                    dbc.Col(html.Div(dcc.Graph(id='GRAPH-CONTENT-3',responsive=False)), width=9),  
                    dbc.Col(html.Div(dcc.RadioItems(options = radio_yfi_list_tab3, value = 'Open Price',id = 'RADIO-SELECTION-3',
                                        style={'textAlign':'center', 'font-size':40}),
                                style={'border':'2px white solid', 'height':'100%'}), width=3)]),
            dcc.Slider(0,5,step=None,marks=slider_date_yfi_dict, value=0,id='SLIDER-SELECTION-3')
                    ]),
               ])
    ])



### Function to update the graph on tab 1, which visualizes different aspects of the stocks
@callback(
    Output('GRAPH-CONTENT', 'figure'),
    Input('DROPDOWN-SELECTION', 'value'),
    Input('SLIDER-SELECTION', 'value'),
    Input('RADIO-SELECTION','value')
)

def update_graph(dropdown_selection, slider_selection, radio_selection):
    plt.close()
    if dropdown_selection in stock_list:
        if radio_selection in ['Open Price','Volume of Shares Sold']:
            radio_selection_conv = radio_yfi_data_dict[radio_selection]
            dff = yfi.Ticker(dropdown_selection).history(period = slider_date_yfi_dict[slider_selection],
                                                interval=slider_date_yfi_interval_dict[slider_selection][0])

            ### Need to change the dates to strings to have the plotting not include days don't have data for and it creates
            ### a long gap. THIS COULD BE SPED UP BY SOME OTHER METHOD
            i=0
            date_str_list = []
            for a in dff.index:
                date_str_list.append(str(dff.index[i]).split(' ')[0] + '\n' + str(dff.index[i]).split(' ')[1])
                i+=1
                
            dff['date_str'] = date_str_list
            dff = dff.reset_index()
            dff = dff.set_index('date_str')
            
            fig, ax = plt.subplots(1,1, figsize=(14,5))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.plot(dff.index, dff[radio_yfi_data_dict[radio_selection]])
            
            plt.title(dropdown_selection + ' - ' + slider_date_yfi_interval_dict[slider_selection][1], 
                     fontsize=20)
            plt.ylabel(radio_selection, fontsize=16)
            
            plt.xticks(fontsize=8, rotation=45)
            plt.yticks(fontsize=14)
            plt.grid()
            html_matplot = plotly.tools.mpl_to_plotly(fig)
            return html_matplot
        if radio_selection == 'Gross Profit':
            dff = yfi.Ticker(dropdown_selection).financials.loc['Gross Profit']
            
            fig, ax = plt.subplots(1,1, figsize=(14,5))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.plot(dff)
            
            plt.title(dropdown_selection + ' - ' + 'Gross Profit ($)', 
                     fontsize=20)
            plt.ylabel('Gross Profit ($)', fontsize=16)
            plt.xticks(fontsize=8, rotation=45)
            plt.yticks(fontsize=14)
            plt.grid()
            html_matplot = plotly.tools.mpl_to_plotly(fig)
            return html_matplot


# ---------------------------------------------- #
### Durgesh code here for running functions!!! ###
# ---------------------------------------------- #

### THIS IS COMMENTED BECAUSE IT NEEDS INPUTS TO RUN PROPERLY. CODE IT UP A BIT AND THEN UNCOMMENT
### Function to update the graph on tab 1, which visualizes different aspects of the stocks
### Function to update the graph on tab 1, which visualizes different aspects of the stocks
@callback(
    Output('GRAPH-CONTENT-2', 'figure'),
    Input('DROPDOWN-SELECTION-2', 'value'),
    Input('RADIO-SELECTION-2','value'),
    Input('K-SELECTION-2', 'value')
)

def update_graph2(dropdown_selection2, radio_selection2, h_selection2):
    plt.close()
    # print(h_selection2)
    if h_selection2 is None:
        return
    if int(h_selection2) < 1:
        return
    h_selection2 = int(h_selection2)
    if dropdown_selection2 in stock_list:
        data_open = pd.DataFrame(yfi.Ticker(dropdown_selection2).history(period='1y', interval='1d')['Open'])
        if radio_selection2 == ['None']:
            plt.close()
            dff = yfi.Ticker(dropdown_selection2).history(period = '1y', interval='1d')

            ### Need to change the dates to strings to have the plotting not include days don't have data for and it creates
            ### a long gap. THIS COULD BE SPED UP BY SOME OTHER METHOD
            i=0
            date_str_list = []
            for a in dff.index:
                date_str_list.append(str(dff.index[i]).split(' ')[0] + '\n' + str(dff.index[i]).split(' ')[1])
                i+=1
                
            dff['date_str'] = date_str_list
            dff = dff.reset_index()
            dff = dff.set_index('date_str')
            
            fig, ax = plt.subplots(1,1, figsize=(14,5))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.plot(dff.index, dff['Open'])
            
            plt.title(dropdown_selection2 + ' - 1 year', 
                     fontsize=20)
            plt.ylabel('Share Open Price', fontsize=16)
            
            plt.xticks(fontsize=8, rotation=45)
            plt.yticks(fontsize=14)
            plt.grid()
            html_matplot = plotly.tools.mpl_to_plotly(fig)
            return html_matplot
        
        if radio_selection2 in ['ElasticNet','Both']:
            data_open_en = pd.DataFrame(yfi.Ticker(dropdown_selection2).history(period='1y', interval='1d')['Open'])
            data_open_en['Open_1day_delay'] = data_open_en['Open'].shift(1)
            data_open_en['Open_2day_delay'] = data_open_en['Open'].shift(2)
            data_open_en['Open_3day_delay'] = data_open_en['Open'].shift(3)
            data_open_en['Open_4day_delay'] = data_open_en['Open'].shift(4)
            data_open_en['Open_5day_delay'] = data_open_en['Open'].shift(5)
            data_open_en['Open_6day_delay'] = data_open_en['Open'].shift(6)
            data_open_en['Open_7day_delay'] = data_open_en['Open'].shift(7)
            data_open_en['Open_8day_delay'] = data_open_en['Open'].shift(8)
            data_open_en['Open_9day_delay'] = data_open_en['Open'].shift(9)
            data_open_en['Open_10day_delay'] = data_open_en['Open'].shift(10)
        
            data_open_en = data_open_en.dropna()
            train_df_x = data_open_en.drop(['Open'],axis=1)
            train_df_y = data_open_en['Open']
        
            alpha = 1
            l1_ratio = 0.12
            model = ElasticNet(alpha=alpha, l1_ratio = l1_ratio)
            model.fit(X=train_df_x, y=train_df_y)

            ### Window for prediction
            pred_list_en = []
            for i in np.arange(0,h_selection2):
                if i == 0:
                    df_conc = pd.DataFrame(np.flip(np.asanyarray(data_open_en['Open'][-1:-11:-1]).reshape(1,-1,)))
                    pred_list_en.append(model.predict(df_conc)[0])
                
                
                if (i > 0) & (i < 11):
                    df_conc = pd.concat([pd.DataFrame(np.flip(np.asanyarray(data_open_en['Open'][-1:min(-11+i,-1):-1]).reshape(1,-1))),
                                                     pd.DataFrame(np.asanyarray(pred_list_en[max(0,i-11):i]).reshape(1,-1))],axis=1)
                    pred_list_en.append(model.predict(df_conc)[0])
                
                if i >= 11:
                    df_conc =  pd.DataFrame(np.asanyarray(pred_list_en[i-10:i+1]).reshape(1,-1))
                    #print(df_conc)
                    pred_list_en.append(model.predict(df_conc)[0])

            to_append_1 = data_open_en['Open'].iloc[-1]
            pred_list_en = np.concatenate([[to_append_1], pred_list_en])
            pred_list_en = pd.DataFrame(pred_list_en)
            pred_list_en.index = pd.date_range(data_open_en.index[-1], periods=h_selection2+1)
            
            en_diff = (pred_list_en.iloc[-1].values - data_open_en.iloc[-1].values) / data_open_en.iloc[-1].values * 100
            en_diff = np.round(en_diff[0],2)


        if radio_selection2 in ['ARIMA','Both']:
            data_open_arima = pd.DataFrame(yfi.Ticker(dropdown_selection2).history(period='1y',interval='1d')['Open'])
            auto = auto_arima(data_open_arima)
            model = ARIMA(data_open_arima, order = auto.order)
            model_fit = model.fit()
            # pred_list_arima = pd.DataFrame(model_fit.forecast(steps=h_selection2))

            pred_list_arima = pd.DataFrame(np.concatenate([data_open_arima.iloc[-1].values,model_fit.forecast(steps=h_selection2)]))
            arima_diff = (pred_list_arima.iloc[-1].values - data_open.iloc[-1].values) / data_open.iloc[-1].values * 100
            arima_diff = np.round(arima_diff[0],2)
            pred_list_arima.index = pd.date_range(data_open_arima.index[-1], periods=h_selection2+1)

        ### graphing !!! 
        dff = data_open.copy()
        
        ### Need to change the dates to strings to have the plotting not include days don't have data for and it creates
        ### a long gap. THIS COULD BE SPED UP BY SOME OTHER METHOD
        i=0
        date_str_list = []
        for a in dff.index:
            date_str_list.append(str(dff.index[i]).split(' ')[0] + '\n' + str(dff.index[i]).split(' ')[1])
            i+=1
            
        dff['date_str'] = date_str_list
        dff = dff.reset_index()
        dff = dff.set_index('date_str')

        plt.close()
        fig, ax = plt.subplots(figsize=(14,5))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.plot(data_open['Open'], label = 'Historical Share Price')
        if radio_selection2 in ['ElasticNet','Both']:
            ax.plot(pred_list_en, label = 'ElasticNet Forecast')
        if radio_selection2 in ['ARIMA','Both']:
            ax.plot(pred_list_arima, label = 'ARIMA Forecast')
        ax.legend(fontsize=12)
        ax.set_ylabel('Share Price')
        ax.set_title('Forecast of ' +  dropdown_selection2 + ' - ' + str(h_selection2) + ' days', fontsize=20)

        if radio_selection2 == 'Both':
            textstr = 'ARIMA predicted differece: ' + str(arima_diff) + '%\nElasticNet predicted difference: ' + str(en_diff) + '%'
        if radio_selection2 == 'ElasticNet':
            textstr = 'ElasticNet predicted difference: ' + str(en_diff) + '%'
        if radio_selection2 == 'ARIMA':
            textstr = 'ARIMA predicted differece: ' + str(arima_diff) + '%'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        if radio_selection2 != 'None':
            ax.text(-0.15, 1.075, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        html_matplot = plotly.tools.mpl_to_plotly(fig)
        return html_matplot


# ---------------------------------------------- #
### Michael code here for running functions!!! ###
### COPY MY GRAPH AND INSTEAD OF PLOT MY DATA PLOT THE DECOMPOSITION ###
# ---------------------------------------------- #


### Function to update the graph on tab 3, which visualizes different aspects of the stock decomposition
@callback(
    Output('GRAPH-CONTENT-3', 'figure'),
    Input('DROPDOWN-SELECTION-3', 'value'),
    Input('SLIDER-SELECTION-3', 'value'),
    Input('RADIO-SELECTION-3','value')
)

def update_graph_3(dropdown_selection3, slider_selection3, radio_selection3):
    plt.close()
    if dropdown_selection3 in stock_list:
        radio_selection_conv = radio_yfi_data_dict[radio_selection3]
        dff = yfi.Ticker(dropdown_selection3).history(period = slider_date_yfi_dict[slider_selection3],
                                            interval=slider_date_yfi_interval_dict[slider_selection3][0])

        
        if slider_selection3 == 5:
            result = seasonal_decompose(dff[radio_selection_conv], model='multiplicative',period = 250)
        if slider_selection3 == 4:
            result = seasonal_decompose(dff[radio_selection_conv], model='multiplicative',period = 12)
        if (slider_selection3 == 3) & (radio_selection_conv == 'Open'):
            result = seasonal_decompose(dff[radio_selection_conv], model='multiplicative',period = 125)   
        if (slider_selection3 == 3) & (radio_selection_conv == 'Volume'):
            result = seasonal_decompose(dff[radio_selection_conv], model='additive',period = 65) 
        if slider_selection3 == 2:
            result = seasonal_decompose(dff[radio_selection_conv], model='multiplicative',period = 65)
        if slider_selection3 == 1:
            result = seasonal_decompose(dff[radio_selection_conv], model='multiplicative',period = 12)
        if (slider_selection3 == 0) & (radio_selection_conv == 'Open'):
            result = seasonal_decompose(dff[radio_selection_conv], model='multiplicative',period = 12)
        if (slider_selection3 == 0) & (radio_selection_conv == 'Volume'):
            result = seasonal_decompose(dff[radio_selection_conv], model='additive',period = 12) 
    
        ### Need to change the dates to strings to have the plotting not include days don't have data for and it creates
        ### a long gap. THIS COULD BE SPED UP BY SOME OTHER METHOD
        i=0
        date_str_list = []
        for a in dff.index:
            date_str_list.append(str(dff.index[i]).split(' ')[0] + '\n' + str(dff.index[i]).split(' ')[1])
            i+=1
            
        dff['date_str'] = date_str_list
        dff = dff.reset_index()
        dff = dff.set_index('date_str')
        fig = result.plot()      
        plt.rcParams["figure.figsize"] = (14,6)        
        title_str = dropdown_selection3 + ' - ' + slider_date_yfi_interval_dict[slider_selection3][1]
        fig.suptitle(title_str, fontsize=20, x=0.2)
        html_matplot = plotly.tools.mpl_to_plotly(fig)
        return html_matplot

        server=app.server

if __name__ == '__main__':
    app.run(debug=True, jupyter_mode='tab')



# %%



