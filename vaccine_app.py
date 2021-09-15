### import libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pmdarima as pm

### load data from Johns Hopkins github repository
death_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
country_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv')

country_vax_df = pd.read_csv('https://raw.githubusercontent.com/govex/COVID-19/master/data_tables/vaccine_data/global_data/vaccine_data_global.csv')
global_vax_admin_df = pd.read_csv('https://raw.githubusercontent.com/govex/COVID-19/master/data_tables/vaccine_data/global_data/time_series_covid19_vaccine_doses_admin_global.csv')
global_vax_full_df = pd.read_csv('https://raw.githubusercontent.com/govex/COVID-19/master/data_tables/vaccine_data/global_data/time_series_covid19_vaccine_global.csv')

### load population data from local csv
pop_raw = pd.read_csv("data/population_by_country_2020.csv")

### data cleaning
# renaming the df column names to lowercase
country_df.columns = map(str.lower, country_df.columns)
confirmed_df.columns = map(str.lower, confirmed_df.columns)
death_df.columns = map(str.lower, death_df.columns)
recovered_df.columns = map(str.lower, recovered_df.columns)

# changing province/state to state and country/region to country
confirmed_df = confirmed_df.rename(columns={'province/state': 'state', 'country/region': 'country'})
recovered_df = confirmed_df.rename(columns={'province/state': 'state', 'country/region': 'country'})
death_df = death_df.rename(columns={'province/state': 'state', 'country/region': 'country'})
country_df = country_df.rename(columns={'country_region': 'country'})

country_vax_df = country_vax_df.rename(columns={'Province_State': 'state', 'Country_Region': 'country'})
global_vax_admin_df = global_vax_admin_df.rename(columns={'Province_State': 'state', 'Country_Region': 'country'})
global_vax_full_df = global_vax_full_df.rename(columns={'Province_State': 'state', 'Country_Region': 'country'})

# merge & drop duplicates to get lat & long for each country vax data
full_country_df = pd.merge(country_df, country_vax_df, on='country')
full_country_df = full_country_df.drop_duplicates(subset= ['country'])
full_country_df['People_fully_vaccinated'] = full_country_df['People_fully_vaccinated'].fillna(0)

### dataframe has country data as well as that country split into states
# drop the ones with states
global_vax_full_df = global_vax_full_df.drop(global_vax_full_df[pd.notna(global_vax_full_df['state'])].index)

# world vax data time series
world_vax_ts = global_vax_full_df[global_vax_full_df['country'] == 'World']

# population data
pop_raw = pop_raw.rename(columns={"Country(or dependency)": "country", "Population(2020)": "population"})
pop = pop_raw[['country', 'population']]
global_vax_full_df= pd.merge(global_vax_full_df, pop, on='country', how = 'left')

full_country_df = pd.merge(full_country_df, pop, on='country', how = 'left')
full_country_df['Percent_fully_vaccinated'] = (full_country_df['People_fully_vaccinated']/full_country_df['population'])
full_country_df['Percent_fully_vaccinated'] = full_country_df['Percent_fully_vaccinated'].fillna(0)

### calculations
# total number of confirmed, death and recovered cases
confirmed_total = int(country_df['confirmed'].sum())
deaths_total = int(country_df['deaths'].sum())
recovered_total = int(country_df['recovered'].sum())
active_total = int(country_df['active'].sum())

doses_admin_total = country_vax_df.loc[country_vax_df['country'] == 'World']['Doses_admin']
full_vax_total = country_vax_df.loc[country_vax_df['country'] == 'World']['People_fully_vaccinated']

### country daily data
# doses
global_vax_full_df['Doses_daily'] = global_vax_full_df.groupby(['country'])['Doses_admin'].transform(lambda s: s.sub(s.shift().fillna(0)).abs())
# full vax
global_vax_full_df['fully_vaccinated_daily'] = global_vax_full_df.groupby(['country'])['People_fully_vaccinated'].transform(lambda s: s.sub(s.shift().fillna(0)).abs())

### world daily data
# doses
world_vax_ts['Doses_daily'] = world_vax_ts['Doses_admin'].transform(lambda s: s.sub(s.shift().fillna(0)).abs())
# full vax
world_vax_ts['fully_vaccinated_daily'] = world_vax_ts['People_fully_vaccinated'].transform(lambda s: s.sub(s.shift().fillna(0)).abs())

world_vax_ts['Percent_fully_vaccinated'] = world_vax_ts['People_fully_vaccinated'] / pop['population'].sum()
world_vax_ts['Percent_fully_daily'] = world_vax_ts['Percent_fully_vaccinated'].transform(lambda s: s.sub(s.shift().fillna(0)).abs())

### world ARIMA modeling % vax
y = world_vax_ts[['Date','Percent_fully_vaccinated']]
val = y['Percent_fully_vaccinated'].values
model = pm.auto_arima(val, start_p=1, start_q=1,
test='adf', # use adftest to find optimal 'd'
max_p=3, max_q=3, # maximum p and q
m=1, # frequency of series
d=None, # let model determine 'd'
seasonal=False, # No Seasonality
start_P=0,
D=0,
trace=True,
error_action='ignore',
suppress_warnings=True,
stepwise=True)

# last update
u1 = country_df.iloc[0]['last_update']
u2 = datetime.strptime(u1, "%Y-%m-%d %H:%M:%S")
update = datetime.strftime(u2, "%b %d %Y %H:%M:%S")

### country covid data
# cases x and y
df_list = [confirmed_df]
for i, df in enumerate(df_list):
    case_x_data = np.array(list(df.iloc[:, 20:].columns))
case_x_data = pd.to_datetime(case_x_data)

def case_y_axis(country, daily=0):
    df_list = [confirmed_df]
    for i, df in enumerate(df_list):
        case_y_data = np.sum(np.asarray(df[df['country'] == country].iloc[:, 20:]),axis = 0)

        new = [0]
        for i in range(1,len(case_y_data)):
            x = case_y_data[i] - case_y_data[i-1]
            new.append(x)   

    return new if daily == 1 else case_y_data

# deaths x and y
df_list = [death_df]
for i, df in enumerate(df_list):
    death_x_data = np.array(list(df.iloc[:, 20:].columns))
death_x_data = pd.to_datetime(death_x_data)

def death_y_axis(country, daily=0):
    df_list = [death_df]
    for i, df in enumerate(df_list):
        death_y_data = np.sum(np.asarray(df[df['country'] == country].iloc[:, 20:]),axis = 0)

        new = [0]
        for i in range(1,len(death_y_data)):
            x = death_y_data[i] - death_y_data[i-1]
            new.append(x)   

    return new if daily == 1 else death_y_data

### worldwide covid data
# cases x and y
df_list = [confirmed_df]
for i, df in enumerate(df_list):
    total_case_x_data = np.array(list(df.iloc[:, 20:].columns))
total_case_x_data = pd.to_datetime(total_case_x_data)

def total_case_y_axis(daily=0):
    df_list = [confirmed_df]
    for i, df in enumerate(df_list):
        total_case_y_data = np.sum(np.asarray(df.iloc[:, 20:]),axis = 0)

        new = [0]
        for i in range(1,len(total_case_y_data)):
            x = total_case_y_data[i] - total_case_y_data[i-1]
            new.append(x)   

    return new if daily == 1 else total_case_y_data

# deaths x and y
df_list = [death_df]
for i, df in enumerate(df_list):
    total_death_x_data = np.array(list(df.iloc[:, 20:].columns))
total_death_x_data = pd.to_datetime(total_death_x_data)

def total_death_y_axis(daily=0):
    df_list = [death_df]
    for i, df in enumerate(df_list):
        total_death_y_data = np.sum(np.asarray(df.iloc[:, 20:]),axis = 0)

        new = [0]
        for i in range(1,len(total_death_y_data)):
            x = total_death_y_data[i] - total_death_y_data[i-1]
            new.append(x)   

    return new if daily == 1 else total_death_y_data

### app colors. Dark blue and grey color scheme
colors = {
    'background': '#1b1f34',
    'charts': '#242a44',
    'text': '#c6c6c6'
}

### get dropdown options for country select
def get_options(list_countries):
    dict_list = []
    for i in list_countries:
        dict_list.append({'label': i, 'value': i})
    return dict_list
    
# clean country list
country_list = country_vax_df.drop(country_vax_df[pd.notna(country_vax_df['state'])].index)
country_list = country_list[country_list['country'] != 'World']
country_list = country_list[country_list['country'] != 'Kosovo']
country_list = country_list[country_list['country'] != 'US (Aggregate)']

####################################### app UI #######################################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
server = app.server

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

navbar = dbc.Navbar(
    [
        html.A(
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("COVID-19 Vaccination Dashboard", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
        ),
    ],
    color="dark",
    dark=True,
)

app.layout = html.Div(
    style={'backgroundColor': colors['background']},
    children=[
        html.Div([navbar]),
        
        html.Div(children=[

            dbc.Card(
                # last update datetime
                dbc.CardBody(
                    [
                        html.P("Last Update:", className="card-title"),
                        html.P(
                            str(update)
                        ),
                    ]
                ),
                style={'height': 100, 'width': 230, 'backgroundColor': colors['charts'], 'color': colors['text'], 'margin': 5, 'text-indent': 5, 'display': 'inline-block'},
            ),
            dbc.Card(
                # world dose data today
                dbc.CardBody(
                    [
                        html.H4("{:,}".format(int(doses_admin_total)), className="card-title"),
                        html.H5(
                            "Doses"
                        ),
                    ]
                ),
                style={'width': 285, 'backgroundColor': colors['charts'], 'color': 'lightgreen', 'margin': 5, 'text-indent': 5, 'display': 'inline-block'},
            ),
            dbc.Card(
                # world fully vax data today
                dbc.CardBody(
                    [
                        html.H4("{:,}".format(int(full_vax_total)), className="card-title"),
                        html.H5(
                            "Fully Vaccinated"
                        ),
                    ]
                ),
                style={'width': 285, 'backgroundColor': colors['charts'], 'color': 'lightgreen', 'margin': 5, 'text-indent': 5, 'display': 'inline-block'},
            ), 
            dbc.Card(
                # world case data today
                dbc.CardBody(
                    [
                        html.H4("{:,}".format(confirmed_total), className="card-title"),
                        html.H5(
                            "Cases"
                        ),
                    ]
                ),
                style={'width': 285, 'backgroundColor': colors['charts'], 'color': 'red', 'margin': 5, 'text-indent': 5, 'display': 'inline-block'},
            ),
            dbc.Card(
                # world death data today
                dbc.CardBody(
                    [
                        html.H4("{:,}".format(deaths_total), className="card-title"),
                        html.H5(
                            "Deaths"
                        ),
                    ]
                ),
                style={'width': 285, 'backgroundColor': colors['charts'], 'color': 'red', 'margin': 5, 'text-indent': 5, 'display': 'inline-block'},
            ),             
            ]),        

        html.Div(style = {'backgroundColor': colors['charts'], 'margin': 5, 'vertical-align': 'top', 'width': 300, 'height': 600, 'display': 'inline-block'},
            children = 
            [  
                # select data for bar & map
                dcc.Dropdown(style = {'width': 300, 'backgroundColor': colors['charts'], 'color': colors['text']},
                    id= 'dataselect',
                    options= [
                        {'label': 'Doses', 'value': 'Doses_admin'},
                        {'label': 'People Fully Vaccinated', 'value': 'People_fully_vaccinated'},
                        {'label': 'Percent Fully Vaccinated', 'value': 'Percent_fully_vaccinated'},
                        {'label': 'Cases', 'value': 'confirmed'},
                        {'label': 'Deaths', 'value': 'deaths'}
                    ],
                    placeholder="Select the data",
                    value='Doses_admin',
                    clearable=False,
                    multi=False,
                ),
                html.P("No. of Countries to Show: ", style={'margin': 0, 'font-size': 14, 'width': 160, 'display': 'inline-block', 'color': colors['text'], 'textAlign': 'center'}
                ),
                # input number for bar chart
                dcc.Input(style = {'margin': 0, 'width': 140, 'backgroundColor': colors['charts'], 'color': colors['text'], 'border-color': colors['text'],},
                    id="countrynumberselect", type="number", placeholder="Country Number", value = 20
                ),
                # bar chart
                dcc.Graph(id= 'bar',
                style = {'vertical-align': 'top', 'margin-top': 0, 'width': 300, 'height': 539, 'display': 'inline-block'}
                ),    
            ]),

        html.Div(style = {'margin': 5, 'vertical-align': 'top', 'width': 700, 'height': 605, 'display': 'inline-block'}, children=
            [                       
                 # map plot 
                dcc.Graph(id= 'map',
                style = {'vertical-align': 'top', 'height': 605, 'display': 'inline-block'}
                ), 
            ]),

        html.Div(style = {'backgroundColor': colors['charts'], 'margin': 5, 'width': 390, 'display': 'inline-block'},
            children = 
            [  
                # world total area chart
                dcc.Graph(id = 'world_area', style = {'height': 297, 'vertical-align': 'bottom', 'backgroundColor': colors['charts'], 'color': colors['text']}
                ),
                html.Hr(style = {'height': 10, 'width': 390, 'margin-top': 0, 'margin-bottom': 0, 'backgroundColor': colors['background']}
                ),
                # world daily area chart
                dcc.Graph(id = 'world_daily', style = {'height': 297, 'vertical-align': 'bottom', 'backgroundColor': colors['charts'], 'color': colors['text']}
                ),
            ]),
        
        html.Div(style={'height': 305, 'margin': 5, 'margin-top': 0,'vertical-align': 'top', 'display': 'inline-block'}, children=
            [                       
                # scatter plot 
                dcc.Graph(id='scatter1', style={'height': 305, 'width': 695, 'margin-right': 5, 'display': 'inline-block'},
                ),
                # stacked bar chart
                dcc.Graph(id='stackbar', style={'height': 305, 'width': 705, 'margin-left': 5, 'display': 'inline-block'},
                ),
            ]),

        html.Div(style = {'backgroundColor': colors['charts'], 'margin': 5, 'margin-bottom': 0, 'vertical-align': 'top', 'width': 1410, 'height': 255, 'display': 'inline-block'}, children=
            [                       
                html.P("No. of Days to Forecast: ", style={'font-size': 14, 'width': 160, 'display': 'inline-block', 'color': colors['text'], 'textAlign': 'center'}
                ),
                # input number for arima forecast of world % vax
                dcc.Input(style = {'width': 100, 'backgroundColor': colors['charts'], 'color': colors['text'], 'border-color': colors['text'],},
                    id="numberselect", type="number", placeholder="Forecast", value = 50
                ),
                # world % vax with arima 
                dcc.Graph(id = '% vax', style = {'height': 200, 'vertical-align': 'bottom', 'backgroundColor': colors['charts'], 'color': colors['text']}
                ),
            ]),

        html.Div(style = {'margin': 5, 'height': 245, 'width': 305, 'vertical-align': 'top',  'display': 'inline-block'}, children = 
            [
                # below map & scatter. Left side
                # horizontal rule and Filter title
                html.Hr(style = {'height': 5, 'width': 305, 'margin-top': 5, 'margin-bottom': 0, 'backgroundColor': colors['charts']}
                ),
                html.H4("FILTERS", style={'margin-bottom': 15, 'text-indent': 0, 'color': colors['text']}
                ),
                # select the countries dropdown
                html.P("Country:", style={'margin-bottom': 0, 'text-indent': 0, 'color': colors['text']}
                ),
                dcc.Dropdown(style = {'width': 305, 'margin-bottom': 15, 'backgroundColor': colors['charts'], 'color': colors['text']},
                    id= 'countryselect',
                    options= get_options(country_list['country'].unique()),
                    placeholder="Select a country",
                    value=['US', 'China', 'Brazil', 'India'],
                    clearable=False,
                    multi=True,
                ),
                # select the metric dropdown
                html.P("Metric:", style={'margin-bottom': 0, 'text-indent': 0, 'color': colors['text']}
                ),
                dcc.Dropdown(style = {'width': 305, 'margin-bottom': 15, 'backgroundColor': colors['charts'], 'color': colors['text']},
                    id = 'metricselect',
                    options=[
                        {'label': 'Cumulative', 'value': 'Cumulative'},
                        {'label': 'Daily', 'value': 'Daily'},
                        {'label': 'Cumulative Per Capita', 'value': 'Cumulative Per Capita'},
                        {'label': 'Daily Per Capita', 'value': 'Daily Per Capita'}
                    ],
                    placeholder="Select a metric",
                    value = 'Cumulative',
                    clearable=False,
                    multi=False
                ),
                # select the smoothing dropdown
                html.P("Smoothing:", style={'margin-bottom': 0, 'text-indent': 0, 'color': colors['text']}
                ),
                dcc.Dropdown(style = {'width': 305, 'backgroundColor': colors['charts'], 'color': colors['text']},
                    id = 'smoothselect',
                    options=[
                        {'label': 'No Smoothing', 'value': 'No Smoothing'},
                        {'label': '7 Day Moving Average', 'value': '7 Day Moving Average'},
                        {'label': '30 Day Moving Average', 'value': '30 Day Moving Average'}
                    ],
                    placeholder="Select a smoothing option",
                    value = 'No Smoothing',
                    clearable=False,
                    multi=False
                ),
            ]),

        html.Div(style = {'margin': 5, 'height': 270, 'width': 1095, 'vertical-align': 'top', 'display': 'inline-block'}, children = 
            [
                # world doses & fully vax
                dcc.Graph(id= 'world_vax', style = {'height': 270, 'width': 540, 'margin-right': 10, 'margin-top': 5, 'backgroundColor': colors['charts'], 'color': colors['text'], 'display': 'inline-block'}
                ),
                # world covid cases & deaths
                dcc.Graph(id= 'world_covid', style = {'height': 270, 'width': 545, 'margin-top': 5, 'backgroundColor': colors['charts'], 'color': colors['text'], 'display': 'inline-block'}
                ),
            ]),

        html.Div(style = {'margin': 5, 'margin-top': 10, 'width': 695, 'display': 'inline-block'}, children = 
            [   
                # Left side
                # country doses
                dcc.Graph(id= 'dose', style={'height': 400, 'width': 695, 'margin-bottom': 10}
                ),  
                # country covid cases
                dcc.Graph(id= 'case', style={'height': 400, 'width': 695}
                ),
            ]),
        
        html.Div(style = {'margin': 5, 'width': 705, 'display': 'inline-block'}, children = 
            [
                # Right side
                # country full vax
                dcc.Graph(id= 'fullvax', style={'height': 400, 'width': 705, 'margin-bottom': 10}
                ), 
                # country covid deaths
                dcc.Graph(id= 'death', style={'height': 400, 'width': 705}
                ),
            ]),

            ]
    )

####################################### interactive components ####################################### 
# doses chart
@app.callback(Output('dose', 'figure'),
              Input('countryselect', 'value'),
              Input('metricselect', 'value'),
              Input('smoothselect', 'value'))
def update_dose(countryselect, metricselect, smoothselect):
    trace = []  
    df_sub = global_vax_full_df
    # add lines for each country
    for country in countryselect:   
        if metricselect == 'Cumulative' and smoothselect == 'No Smoothing':
            ydata = df_sub[df_sub['country'] == country]['Doses_admin']
        elif metricselect == 'Cumulative' and smoothselect == '7 Day Moving Average':
            ydata = df_sub[df_sub['country'] == country]['Doses_admin'].rolling(window=7).sum()
        elif metricselect == 'Cumulative' and smoothselect == '30 Day Moving Average':
            ydata = df_sub[df_sub['country'] == country]['Doses_admin'].rolling(window=30).sum()
        elif metricselect == 'Cumulative Per Capita' and smoothselect == 'No Smoothing':
            ydata = (df_sub[df_sub['country'] == country]['Doses_admin']/df_sub[df_sub['country'] == country]['population'])*100
        elif metricselect == 'Cumulative Per Capita' and smoothselect == '7 Day Moving Average':
            ydata = ((df_sub[df_sub['country'] == country]['Doses_admin']/df_sub[df_sub['country'] == country]['population'])*100).rolling(window=7).sum()
        elif metricselect == 'Cumulative Per Capita' and smoothselect == '30 Day Moving Average':
            ydata = ((df_sub[df_sub['country'] == country]['Doses_admin']/df_sub[df_sub['country'] == country]['population'])*100).rolling(window=30).sum()
        elif metricselect == 'Daily' and smoothselect == 'No Smoothing':
            ydata = df_sub[df_sub['country'] == country]['Doses_daily']  
        elif metricselect == 'Daily' and smoothselect == '7 Day Moving Average':
            ydata = df_sub[df_sub['country'] == country]['Doses_daily'].rolling(window=7).sum()
        elif metricselect == 'Daily' and smoothselect == '30 Day Moving Average':
            ydata = df_sub[df_sub['country'] == country]['Doses_daily'].rolling(window=30).sum()  
        elif metricselect == 'Daily Per Capita' and smoothselect == 'No Smoothing':
            ydata = (df_sub[df_sub['country'] == country]['Doses_daily']/df_sub[df_sub['country'] == country]['population'])*100
        elif metricselect == 'Daily Per Capita' and smoothselect == '7 Day Moving Average':
            ydata = ((df_sub[df_sub['country'] == country]['Doses_daily']/df_sub[df_sub['country'] == country]['population'])*100).rolling(window=7).sum() 
        elif metricselect == 'Daily Per Capita' and smoothselect == '30 Day Moving Average':
            ydata = ((df_sub[df_sub['country'] == country]['Doses_daily']/df_sub[df_sub['country'] == country]['population'])*100).rolling(window=30).sum()

        trace.append(go.Scatter(x=df_sub[df_sub['country'] == country]['Date'],
                                y=ydata,
                                mode='lines',
                                opacity=0.7,
                                name=country,
                                textposition='bottom center'))   
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # figure layout
    figure = {'data': data,
              'layout': go.Layout(
                template='plotly_dark',
                plot_bgcolor=colors['charts'],
                paper_bgcolor=colors['charts'],
                font_color=colors['text'],
                margin = dict(l = 5, r = 7, t = 25, b = 5),
                hovermode='x',
                autosize=True,
                title={'text': 'Vaccine Doses Administered', 'font_size': 12},
              ),
              }
    return figure

# fully vax chart
@app.callback(Output('fullvax', 'figure'),
             Input('countryselect', 'value'),
             Input('metricselect', 'value'),
             Input('smoothselect', 'value'))
def update_fullvax(countryselect, metricselect, smoothselect):
    trace = []  
    df_sub = global_vax_full_df
    # add lines for each country
    for country in countryselect: 
        if metricselect == 'Cumulative' and smoothselect == 'No Smoothing':
            ydata = df_sub[df_sub['country'] == country]['People_fully_vaccinated']
        elif metricselect == 'Cumulative' and smoothselect == '7 Day Moving Average':
            ydata = df_sub[df_sub['country'] == country]['People_fully_vaccinated'].rolling(window=7).sum()
        elif metricselect == 'Cumulative' and smoothselect == '30 Day Moving Average':
            ydata = df_sub[df_sub['country'] == country]['People_fully_vaccinated'].rolling(window=30).sum()
        elif metricselect == 'Cumulative Per Capita' and smoothselect == 'No Smoothing':
            ydata = (df_sub[df_sub['country'] == country]['People_fully_vaccinated']/df_sub[df_sub['country'] == country]['population'])*100
        elif metricselect == 'Cumulative Per Capita' and smoothselect == '7 Day Moving Average':
            ydata = ((df_sub[df_sub['country'] == country]['People_fully_vaccinated']/df_sub[df_sub['country'] == country]['population'])*100).rolling(window=7).sum()
        elif metricselect == 'Cumulative Per Capita' and smoothselect == '30 Day Moving Average':
            ydata = ((df_sub[df_sub['country'] == country]['People_fully_vaccinated']/df_sub[df_sub['country'] == country]['population'])*100).rolling(window=30).sum()
        elif metricselect == 'Daily' and smoothselect == 'No Smoothing':
            ydata = df_sub[df_sub['country'] == country]['fully_vaccinated_daily']  
        elif metricselect == 'Daily' and smoothselect == '7 Day Moving Average':
            ydata = df_sub[df_sub['country'] == country]['fully_vaccinated_daily'].rolling(window=7).sum()
        elif metricselect == 'Daily' and smoothselect == '30 Day Moving Average':
            ydata = df_sub[df_sub['country'] == country]['fully_vaccinated_daily'].rolling(window=30).sum()    
        elif metricselect == 'Daily Per Capita' and smoothselect == 'No Smoothing':
            ydata = (df_sub[df_sub['country'] == country]['fully_vaccinated_daily']/df_sub[df_sub['country'] == country]['population'])*100 
        elif metricselect == 'Daily Per Capita' and smoothselect == '7 Day Moving Average':
            ydata = ((df_sub[df_sub['country'] == country]['fully_vaccinated_daily']/df_sub[df_sub['country'] == country]['population'])*100).rolling(window=7).sum()
        elif metricselect == 'Daily Per Capita' and smoothselect == '30 Day Moving Average':
            ydata = ((df_sub[df_sub['country'] == country]['fully_vaccinated_daily']/df_sub[df_sub['country'] == country]['population'])*100).rolling(window=30).sum()

        trace.append(go.Scatter(x=df_sub[df_sub['country'] == country]['Date'],
                                y=ydata,
                                mode='lines',
                                opacity=0.7,
                                name=country,
                                textposition='bottom center'))    
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # figure layout
    figure = {'data': data,
              'layout': go.Layout(
                template='plotly_dark',
                plot_bgcolor=colors['charts'],
                paper_bgcolor=colors['charts'],
                font_color=colors['text'],
                margin = dict(l = 5, r = 7, t = 25, b = 5),
                hovermode='x',
                autosize=True,
                title={'text': 'People Fully Vaccinated', 'font_size': 12},
              ),
              }
    return figure

# cases chart
@app.callback(Output('case', 'figure'),
              Input('countryselect', 'value'),
              Input('metricselect', 'value'),
              Input('smoothselect', 'value'))
def update_cases(countryselect, metricselect, smoothselect):
    trace = []  
    # add lines for each country
    for country in countryselect: 
        if metricselect == 'Cumulative' and smoothselect == 'No Smoothing':
            ydata = case_y_axis(country) 
        elif metricselect == 'Cumulative' and smoothselect == '7 Day Moving Average':
            ydata = np.convolve(case_y_axis(country), np.ones(7)/7, mode='valid')
        elif metricselect == 'Cumulative' and smoothselect == '30 Day Moving Average':
            ydata = np.convolve(case_y_axis(country), np.ones(30)/30, mode='valid')
        elif metricselect == 'Cumulative Per Capita' and smoothselect == 'No Smoothing':
            ydata = (case_y_axis(country)/int(pop[pop['country']==country]['population']))*100
        elif metricselect == 'Cumulative Per Capita' and smoothselect == '7 Day Moving Average':
            ydata = (np.convolve(case_y_axis(country), np.ones(7)/7, mode='valid')/int(pop[pop['country']==country]['population']))*100
        elif metricselect == 'Cumulative Per Capita' and smoothselect == '30 Day Moving Average':
            ydata = (np.convolve(case_y_axis(country), np.ones(30)/30, mode='valid')/int(pop[pop['country']==country]['population']))*100
        elif metricselect == 'Daily' and smoothselect == 'No Smoothing': 
            ydata = case_y_axis(country,1) 
        elif metricselect == 'Daily' and smoothselect == '7 Day Moving Average':
            ydata = np.convolve(case_y_axis(country,1), np.ones(7)/7, mode='valid')
        elif metricselect == 'Daily' and smoothselect == '30 Day Moving Average':
            ydata = np.convolve(case_y_axis(country,1), np.ones(30)/30, mode='valid')
        elif metricselect == 'Daily Per Capita' and smoothselect == 'No Smoothing': 
            ydata = [(i/int(pop[pop['country']==country]['population']))*100 for i in case_y_axis(country,1)]
        elif metricselect == 'Daily Per Capita' and smoothselect == '7 Day Moving Average':
            ydata = np.convolve([(i/int(pop[pop['country']==country]['population']))*100 for i in case_y_axis(country,1)], np.ones(7)/7, mode='valid')
        elif metricselect == 'Daily Per Capita' and smoothselect == '30 Day Moving Average':
            ydata = np.convolve([(i/int(pop[pop['country']==country]['population']))*100 for i in case_y_axis(country,1)], np.ones(30)/30, mode='valid')
  
        trace.append(go.Scatter(x=case_x_data,
                                y=ydata,
                                mode='lines',
                                opacity=0.7,
                                name=country,
                                textposition='bottom center'))  
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # figure layout
    figure = {'data': data,
              'layout': go.Layout(
                template='plotly_dark',
                plot_bgcolor=colors['charts'],
                paper_bgcolor=colors['charts'],
                font_color=colors['text'],
                margin = dict(l = 5, r = 7, t = 25, b = 5),
                hovermode='x',
                autosize=True,
                title={'text': 'Covid 19 Cases', 'font_size': 12},
              ),
              }
    return figure

# deaths chart
@app.callback(Output('death', 'figure'),
              Input('countryselect', 'value'),
              Input('metricselect', 'value'),
              Input('smoothselect', 'value'))
def update_deaths(countryselect, metricselect, smoothselect):
    trace = []  
    # add lines for each country
    for country in countryselect:
        if metricselect == 'Cumulative' and smoothselect == 'No Smoothing':
            ydata = death_y_axis(country)
        elif metricselect == 'Cumulative' and smoothselect == '7 Day Moving Average':
            ydata = np.convolve(death_y_axis(country), np.ones(7)/7, mode='valid')
        elif metricselect == 'Cumulative' and smoothselect == '30 Day Moving Average':
            ydata = np.convolve(death_y_axis(country), np.ones(30)/30, mode='valid')
        elif metricselect == 'Cumulative Per Capita' and smoothselect == 'No Smoothing':
            ydata = (death_y_axis(country)/int(pop[pop['country']==country]['population']))*100
        elif metricselect == 'Cumulative Per Capita' and smoothselect == '7 Day Moving Average':
            ydata = (np.convolve(death_y_axis(country), np.ones(7)/7, mode='valid')/int(pop[pop['country']==country]['population']))*100
        elif metricselect == 'Cumulative Per Capita' and smoothselect == '30 Day Moving Average':
            ydata = (np.convolve(death_y_axis(country), np.ones(30)/30, mode='valid')/int(pop[pop['country']==country]['population']))*100
        elif metricselect == 'Daily' and smoothselect == 'No Smoothing':  
            ydata = death_y_axis(country,1)
        elif metricselect == 'Daily' and smoothselect == '7 Day Moving Average':
            ydata = np.convolve(death_y_axis(country,1), np.ones(7)/7, mode='valid')
        elif metricselect == 'Daily' and smoothselect == '30 Day Moving Average':
            ydata = np.convolve(death_y_axis(country,1), np.ones(30)/30, mode='valid')
        elif metricselect == 'Daily Per Capita' and smoothselect == 'No Smoothing':  
            ydata = [(i/int(pop[pop['country']==country]['population']))*100 for i in death_y_axis(country,1)]
        elif metricselect == 'Daily Per Capita' and smoothselect == '7 Day Moving Average':
            ydata = np.convolve([(i/int(pop[pop['country']==country]['population']))*100 for i in death_y_axis(country,1)], np.ones(7)/7, mode='valid')
        elif metricselect == 'Daily Per Capita' and smoothselect == '30 Day Moving Average':
            ydata = np.convolve([(i/int(pop[pop['country']==country]['population']))*100 for i in death_y_axis(country,1)], np.ones(30)/30, mode='valid')
  
        trace.append(go.Scatter(x=death_x_data,
                                y=ydata,
                                mode='lines',
                                opacity=0.7,
                                name=country,
                                textposition='bottom center'))
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # figure layout
    figure = {'data': data,
              'layout': go.Layout(
                template='plotly_dark',
                plot_bgcolor=colors['charts'],
                paper_bgcolor=colors['charts'],
                font_color=colors['text'],
                margin = dict(l = 5, r = 7, t = 25, b = 5),
                hovermode='x',
                autosize=True,
                title={'text': 'Covid 19 Deaths', 'font_size': 12},
              ),
              }
    return figure

# % fully vax with arima forecast
@app.callback(Output('% vax', 'figure'),
             Input('numberselect', 'value'))
def update_percentvax(numberselect):
    # forecast
    n_periods = numberselect
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = np.arange(len(val), len(val)+n_periods)
    # make series for plotting 
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    # add % vax into df
    fc_ts = pd.DataFrame(world_vax_ts['Date'])
    fc_ts['Percent_fully_vaccinated'] = val.tolist()
    # add extra dates for forecast
    test = pd.DataFrame({'Date': pd.date_range(start=fc_ts.Date.iloc[-1], periods=n_periods)})
    test = test.iloc[1: , :]
    test['Date'] = test['Date'].dt.strftime("%Y-%m-%d")
    # add forecast & conf intervals into df
    fc_ts = pd.concat([fc_ts, test])
    fc_ts.reset_index(drop=True, inplace=True)
    fc_ts['forecast'] = fc_series 
    fc_ts['upper'] = upper_series
    fc_ts['lower'] = lower_series
    # add lines to figure
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=fc_ts['Date'], y=fc_ts['Percent_fully_vaccinated'], mode='lines', name='% vaccinated', hovertemplate='Date: %{x} <br>%{y:.2%}', showlegend = False))
    figure.add_trace(go.Scatter(x=fc_ts['Date'], y=fc_ts['forecast'], mode='lines', name='forecast', hovertemplate='Date: %{x} <br>%{y:.2%}', showlegend = False))
    figure.add_trace(go.Scatter(x=fc_ts['Date'], y=fc_ts['upper'], fill='tonexty', mode='lines', name='upper bound', hovertemplate='Date: %{x} <br>%{y:.2%}', showlegend = False))
    figure.add_trace(go.Scatter(x=fc_ts['Date'], y=fc_ts['lower'], fill='tonexty', mode='lines', name='lower bound', hovertemplate='Date: %{x} <br>%{y:.2%}', showlegend = False))
    # figure layout
    figure.update_layout(
                template='plotly_dark',
                plot_bgcolor=colors['charts'],
                paper_bgcolor=colors['charts'],
                font_color=colors['text'],
                margin = dict(l = 5, r = 7, t = 25, b = 5),
                autosize=True,
                title={'text': 'Percent of the World Fully Vaccinated', 'font_size': 12},
              )
    return figure

# world vaccine chart
@app.callback(Output('world_vax', 'figure'),
              Input('metricselect', 'value'),
              Input('smoothselect', 'value'))
def update_world_vax(metricselect, smoothselect):
    trace = []  
    # create lines based on metric
    if (metricselect in ('Cumulative', 'Cumulative Per Capita')) and smoothselect == 'No Smoothing': 
        ydata1 = world_vax_ts['Doses_admin']
        ydata2 = world_vax_ts['People_fully_vaccinated']   
    elif (metricselect in ('Cumulative', 'Cumulative Per Capita')) and smoothselect == '7 Day Moving Average': 
        ydata1 = world_vax_ts['Doses_admin'].rolling(window=7).sum()
        ydata2 = world_vax_ts['People_fully_vaccinated'].rolling(window=7).sum()
    elif (metricselect in ('Cumulative', 'Cumulative Per Capita')) and smoothselect == '30 Day Moving Average': 
        ydata1 = world_vax_ts['Doses_admin'].rolling(window=30).sum()
        ydata2 = world_vax_ts['People_fully_vaccinated'].rolling(window=30).sum()
    elif (metricselect in ('Daily', 'Daily Per Capita')) and smoothselect == 'No Smoothing': 
        ydata1 = world_vax_ts['Doses_daily']
        ydata2 = world_vax_ts['fully_vaccinated_daily']
    elif (metricselect in ('Daily', 'Daily Per Capita')) and smoothselect == '7 Day Moving Average':  
        ydata1 = world_vax_ts['Doses_daily'].rolling(window=7).sum()
        ydata2 = world_vax_ts['fully_vaccinated_daily'].rolling(window=7).sum()
    elif (metricselect in ('Daily', 'Daily Per Capita')) and smoothselect == '30 Day Moving Average':   
        ydata1 = world_vax_ts['Doses_daily'].rolling(window=30).sum()
        ydata2 = world_vax_ts['fully_vaccinated_daily'].rolling(window=30).sum()

    trace.append(go.Scatter(x=world_vax_ts['Date'],
                            y=ydata1,
                            mode='lines',
                            line=dict(color="green"),
                            opacity=0.7,
                            name='Doses',
                            textposition='bottom center')) 
    trace.append(go.Scatter(x=world_vax_ts['Date'],
                            y=ydata2,
                            mode='lines',
                            line=dict(color="lightgreen"),
                            opacity=0.7,
                            name='Fully Vaccinated',
                            textposition='bottom center')) 
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # figure layout
    figure = {'data': data,
              'layout': go.Layout(
                template='plotly_dark',
                plot_bgcolor=colors['charts'],
                paper_bgcolor=colors['charts'],
                font_color=colors['text'],
                margin = dict(l = 5, r = 7, t = 25, b = 5),
                hovermode='x',
                autosize=True,
                showlegend=True,
                title={'text': 'World Vaccine Data', 'font_size': 12},
              ),
              }
    return figure

# world covid chart
@app.callback(Output('world_covid', 'figure'),
              Input('metricselect', 'value'),
              Input('smoothselect', 'value'))
def update_world_covid(metricselect, smoothselect):
    trace = []  
    # create lines based on metric
    if (metricselect in ('Cumulative', 'Cumulative Per Capita')) and smoothselect == 'No Smoothing': 
        ydata1 = total_case_y_axis(0)
        ydata2 = total_death_y_axis(0)
    elif (metricselect in ('Cumulative', 'Cumulative Per Capita')) and smoothselect == '7 Day Moving Average':  
        ydata1 = np.convolve(total_case_y_axis(0), np.ones(7)/7, mode='valid')
        ydata2 = np.convolve(total_death_y_axis(0), np.ones(7)/7, mode='valid') 
    elif (metricselect in ('Cumulative', 'Cumulative Per Capita')) and smoothselect == '30 Day Moving Average':   
        ydata1 = np.convolve(total_case_y_axis(0), np.ones(30)/30, mode='valid')
        ydata2 = np.convolve(total_death_y_axis(0), np.ones(30)/30, mode='valid')
    elif (metricselect in ('Daily', 'Daily Per Capita')) and smoothselect == 'No Smoothing': 
        ydata1 = total_case_y_axis(1)
        ydata2 = total_death_y_axis(1)
    elif (metricselect in ('Daily', 'Daily Per Capita')) and smoothselect == '7 Day Moving Average':   
        ydata1 = np.convolve(total_case_y_axis(1), np.ones(7)/7, mode='valid')
        ydata2 = np.convolve(total_death_y_axis(1), np.ones(7)/7, mode='valid')  
    elif (metricselect in ('Daily', 'Daily Per Capita')) and smoothselect == '30 Day Moving Average':   
        ydata1 = np.convolve(total_case_y_axis(1), np.ones(30)/30, mode='valid')
        ydata2 = np.convolve(total_death_y_axis(1), np.ones(30)/30, mode='valid')

    trace.append(go.Scatter(x=total_case_x_data,
                            y=ydata1,
                            mode='lines',
                            line=dict(color="red"),
                            opacity=0.7,
                            name='Cases',
                            textposition='bottom center'))
    trace.append(go.Scatter(x=total_death_x_data,
                            y=ydata2,
                            mode='lines',
                            line=dict(color="lightcoral"),
                            opacity=0.7,
                            name='Deaths',
                            textposition='bottom center')) 
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # figure layout
    figure = {'data': data,
              'layout': go.Layout(
                template='plotly_dark',
                plot_bgcolor=colors['charts'],
                paper_bgcolor=colors['charts'],
                font_color=colors['text'],
                margin = dict(l = 5, r = 7, t = 25, b = 5),
                hovermode='x',
                autosize=True,
                showlegend=True,
                title={'text': 'World Covid 19 Data', 'font_size': 12},
              ),
              }
    return figure

# world map chart
@app.callback(Output('map', 'figure'),
              Input('dataselect', 'value'))
def update_world_map(dataselect):
    if (dataselect == 'Doses_admin' or dataselect == 'People_fully_vaccinated' or dataselect == 'Percent_fully_vaccinated'):
        circle_color = px.colors.sequential.Greens
        if dataselect == 'Doses_admin':
            name = 'Doses'
        elif dataselect == 'People_fully_vaccinated':
            name = 'People Fully Vaccinated'
        else:
            name = 'Percent Fully Vaccinated'
    else:
        circle_color = px.colors.sequential.Reds
        if dataselect == 'confirmed':
            name = 'Cases'
        else:
            name = 'Deaths'
    # create map plot
    figure = px.scatter_mapbox(full_country_df, lat="lat", lon="long_", color=dataselect, size=dataselect, 
        size_max=55, hover_name="country", labels = {dataselect: 'Total ' + name}, color_continuous_scale=circle_color, 
        zoom = 1, mapbox_style="carto-darkmatter")
    figure.update_layout(
        title = {'text': 'No. of Total ' + name + ' by Country'},
        title_font_size = 12,
        coloraxis_showscale=False,
        paper_bgcolor=colors['charts'],
        font_color=colors['text'],
        margin = dict(l = 5, r = 5, t = 25, b = 5)
        )

    return figure

# world vertical bar chart
@app.callback(Output('bar', 'figure'),
              Input('dataselect', 'value'),
              Input('countrynumberselect', 'value'))
def update_world_bar(dataselect, countrynumberselect):
    bar_data = full_country_df.sort_values(dataselect, ascending= False)
    # regular bar chart data
    bar_data = bar_data.head(countrynumberselect)
    bar_data = bar_data.iloc[::-1]
    if (dataselect == 'Doses_admin' or dataselect == 'People_fully_vaccinated' or dataselect == 'Percent_fully_vaccinated'):
        bar_color = 'lightgreen'
        if dataselect == 'Doses_admin':
            name = 'Doses'
        elif dataselect == 'People_fully_vaccinated':
            name = 'People Fully Vacinated'
        else:
            name = 'Percent Fully Vaccinated'
    else:
        bar_color = 'red'
        if dataselect == 'confirmed':
            name = 'Cases'
        else:
            name = 'Deaths'
    # create bar plot
    figure = go.Figure(
        data= go.Bar(
            x= bar_data[dataselect],
            y= bar_data['country'],
            orientation='h',
            marker=dict(
            color=bar_color,
            line=dict(color=bar_color, width=1))
            ),
        layout = go.Layout(
                template='plotly_dark',
                plot_bgcolor=colors['charts'],
                paper_bgcolor=colors['charts'],
                font_color=colors['text'],
                margin=dict(
                    l=5,
                    r=5,
                    b=5,
                    t=55,
                    pad=5
                ),
                autosize=True,
                title={'text': 'No. of Total ' + name +  ' by Country', 'font_size': 11},
                )
        )
    return figure

# world single total area chart
@app.callback(Output('world_area', 'figure'),
              Input('dataselect', 'value'))
def update_world_area(dataselect):
    if dataselect == 'Doses_admin':
        name = 'Doses'
        figure = px.area(world_vax_ts, x='Date', y='Doses_admin', color_discrete_sequence=['lightgreen'])
    elif dataselect == 'People_fully_vaccinated':
        name = 'People Fully Vaccinated'
        figure = px.area(world_vax_ts, x='Date', y='People_fully_vaccinated', color_discrete_sequence=['lightgreen'])
    elif dataselect == 'Percent_fully_vaccinated':
        name = 'Percent Fully Vaccinated'
        figure = px.area(world_vax_ts, x='Date', y='Percent_fully_vaccinated', color_discrete_sequence=['lightgreen'])
    elif dataselect == 'confirmed':
        name = 'Cases'
        figure = px.area(x=total_case_x_data, y=total_case_y_axis(0), color_discrete_sequence=['red'])
    elif dataselect == 'deaths':
        name = 'Deaths'
        figure = px.area(x=total_death_x_data, y=total_death_y_axis(0), color_discrete_sequence=['red'])

    # area chart
    figure.update_layout(
        template='plotly_dark',
        title="No. of Total " + name,
        title_font_size= 12,
        xaxis_title=None,
        yaxis_title=None,
        plot_bgcolor=colors['charts'],
        paper_bgcolor=colors['charts'],
        font_color=colors['text'],
        font_size=11,
        margin = dict(l = 5, r = 5, t = 35, b = 5),
        showlegend=False
    )
    return figure

# world single daily area chart
@app.callback(Output('world_daily', 'figure'),
              Input('dataselect', 'value'))
def update_world_daily(dataselect):
    if dataselect == 'Doses_admin':
        name = 'Doses'
        figure = px.area(x=world_vax_ts['Date'], y=world_vax_ts['Doses_daily'], color_discrete_sequence=['lightgreen'])
    elif dataselect == 'People_fully_vaccinated':
        name = 'People Fully Vaccinated'
        figure = px.area(x=world_vax_ts['Date'], y=world_vax_ts['fully_vaccinated_daily'], color_discrete_sequence=['lightgreen'])
    elif dataselect == 'Percent_fully_vaccinated':
        name = 'Percent Fully Vaccinated'
        figure = px.area(x=world_vax_ts['Date'], y=world_vax_ts['Percent_fully_daily'], color_discrete_sequence=['lightgreen'])
    elif dataselect == 'confirmed':
        name = 'Cases'
        figure = px.area(x=total_case_x_data, y=total_case_y_axis(1), color_discrete_sequence=['red'])
    elif dataselect == 'deaths':
        name = 'Deaths'
        figure = px.area(x=total_death_x_data, y=total_death_y_axis(1), color_discrete_sequence=['red'])

    # area chart
    figure.update_layout(
        template='plotly_dark',
        title="No. of Daily " + name,
        title_font_size= 12,
        xaxis_title=None,
        yaxis_title=None,
        plot_bgcolor=colors['charts'],
        paper_bgcolor=colors['charts'],
        font_color=colors['text'],
        font_size=11,
        margin = dict(l = 5, r = 5, t = 35, b = 5),
        showlegend=False
    )
    return figure

# world stacked bar chart
@app.callback(Output('stackbar', 'figure'),
              Input('dataselect', 'value'),
              Input('countrynumberselect', 'value'))
def update_world_stackbar(dataselect, countrynumberselect):
    bar1_data = full_country_df.sort_values(dataselect, ascending= False)
    # stacked bar chart data
    bar1_data = bar1_data.head(countrynumberselect)
    bar1_data = bar1_data.iloc[::-1]

    figure = go.Figure(
            data= [go.Bar(name= 'Fully',
                y= bar1_data['People_fully_vaccinated'],
                x= bar1_data['country'],
                marker=dict(
                color='lightgreen',
                line=dict(color='lightgreen', width=1))
                ),
                go.Bar(name= 'Partially',
                y= bar1_data['People_partially_vaccinated'],
                x= bar1_data['country'],
                marker=dict(
                color='yellow',
                line=dict(color='yellow', width=1))
                )],
            layout = go.Layout(
                    barmode = 'stack',
                    template='plotly_dark',
                    plot_bgcolor=colors['charts'],
                    paper_bgcolor=colors['charts'],
                    font_color=colors['text'],
                    margin=dict(
                        l=5,
                        r=5,
                        b=5,
                        t=35,
                        pad=5
                    ),
                    autosize=True,
                    title={'text': 'No. of Vaccinated People by Country', 'font_size': 12},
                    showlegend=False
                    )
            )
    return figure

# world scatter bar chart
@app.callback(Output('scatter1', 'figure'),
              Input('dataselect', 'value'),
              Input('countrynumberselect', 'value'))
def update_world_scatter(dataselect, countrynumberselect):
    if (dataselect == 'Doses_admin' or dataselect == 'People_fully_vaccinated' or dataselect == 'Percent_fully_vaccinated'):
        if dataselect == 'Doses_admin':
            name = 'Doses'
        elif dataselect == 'People_fully_vaccinated':
            name = 'People Fully Vacinated'
        else:
            name = 'Percent Fully Vaccinated'
    else:
        if dataselect == 'confirmed':
            name = 'Cases'
        else:
            name = 'Deaths'
    # scatter plot data
    sorted_country_df = full_country_df.sort_values(dataselect, ascending= False)
    # create scatter plot
    figure = px.scatter(sorted_country_df.head(countrynumberselect), x="People_fully_vaccinated", y="confirmed", size="confirmed", color="country",
        log_x=True, hover_name="country", size_max=60)
    figure.update_layout(
        template='plotly_dark',
        title="Top " + str(countrynumberselect) +" Countries by " + name,
        title_font_size= 12,
        xaxis_title="Fully Vaccinated People",
        yaxis_title="Cases",
        plot_bgcolor=colors['charts'],
        paper_bgcolor=colors['charts'],
        font_color=colors['text'],
        font_size=11,
        margin = dict(l = 5, r = 5, t = 35, b = 5),
        showlegend=False
    )
    return figure

if __name__ == "__main__":
    app.run_server(debug=True)
