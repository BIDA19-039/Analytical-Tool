#!/usr/bin/env python
# coding: utf-8

# In[5]:



import random
import datetime
import csv
import pandas as pd


# read log file into DataFrame
#logs_df = pd.read_csv('funolympics.csv', sep=',')
#print(logs_df)


# List of possible HTTP status codes
status_codes = [200, 404, 500]

# List of possible HTTP request methods
request_methods = ["GET", "POST"]

# List of possible paths for the Olympics website
paths = ["/", "/athletes", "/sports", "/medals", "/schedule", "/results",
         "/sports/basketball", "/sports/cycling", "/sports/diving", "/sports/gymnastics", "/sports/rowing", "/sports/soccer",
         "/sports/swimming", "/sports/table-tennis", "/sports/tennis", "/sports/track-and-field", "/sports/volleyball",
         "/sports/water-polo", "/sports/wrestling", "/sports/boxing", "/sports/baseball", "/sports/judo", "/sports/shooting", "/medals", "/about"]

# List of possible user agents
user_agents = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
               "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:54.0) Gecko/20100101 Firefox/54.0",
               "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
               "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/54.0",
               "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"]

# List of possible countries
countries = ["United States","South Korea", "Sweden", "India", "United Kingdom", "Canada", "Australia", "Germany", "France", "Spain", "Italy", "Japan", "China", "South Africa"]

# List of possible traffic sources
traffic_sources = ["Reddit", "Twitter", "Facebook", "Instagram", "TikTok", "YouTube", "Email Marketing", "Direct Traffic", "Other"]


# List of possible gender values
genders = ["male", "female", "non-binary"]

# List of possible age ranges
ages = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

# List of possible interests
interests = ["Schedule and results", "Tickets", "Athlete profiles", "Venue information", "News and articles", "Fan discussions", "Olympic history",
             "Live streaming of events"]

# List of possible number of website visits
visits = [1, 2, 3, 4, 5]


# Generate a random web server log entry
def generate_log_entry():
    start_date = datetime.datetime(2022, 1, 1)
    end_date = datetime.datetime.now()
    time_diff = (end_date - start_date).total_seconds()
    random_seconds = random.randint(0, int(time_diff))
    timestamp = (start_date + datetime.timedelta(seconds=random_seconds)).strftime("%d/%b/%Y:%H:%M:%S %z")
    http_method = random.choice(request_methods)
    path = random.choice(paths)
    http_version = "HTTP/1.1"
    status_code = random.choice(status_codes)
    user_agent = random.choice(user_agents)
    country = random.choice(countries)
    ip_address = f"{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"
    traffic_source = random.choice(traffic_sources)
    gender = random.choice(genders)
    age = random.choice(ages)
    interest = random.choice(interests)
    visits = random.randint(1, 10)
    duration = random.randint(5, 600)
    log_entry = [ip_address, timestamp, http_method, path, http_version, str(status_code), str(random.randint(100, 10000)), traffic_source, user_agent, country, gender, age, visits, interest, duration]
    return log_entry



# Generate a web server log file with the specified number of entries and save it in CSV format
def generate_log_file(num_entries):
    column_headings = ['IP Address', 'Timestamp', 'HTTP Method', 'Path', 'HTTP Version', 'Status Code',
                       'Bytes Transferred', 'Traffic Source', 'User Agent', 'Country', 'Gender', 'Age', 'Visits', 'Interest', 'Duration']
    with open("olympicsdata.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(column_headings) # Write the column headings as the first row
        for i in range(num_entries):
            log_entry = generate_log_entry()
            writer.writerow(log_entry)


# Generate a web server log file 
generate_log_file(4898)


# In[6]:


# read log file into DataFrame 
df = pd.read_csv('olympicsdata.csv', sep=',')
#df[['Date', 'Time']] = df['Timestamp'].str.split(' ', 1, expand=True)
df


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[11]:


# Calculate the total visits
total_visits = df['Visits'].sum()
print("Total Visits:", total_visits)


# In[12]:


# Calculate the total visits by gender
gender_visits = df.groupby('Gender')['Visits'].sum()

# Plot the bar graph
plt.figure(figsize=(8, 6))
gender_visits.plot(kind='bar')
plt.title('Total Visits by Gender')
plt.xlabel('Gender')
plt.ylabel('Visits')

plt.xticks(rotation=0)  # Rotate x-axis labels if needed
plt.show()


# In[13]:


# Count the values in the 'Path' column
column1_count = df['Path'].value_counts()

# Plot the count of 'Path'
plt.figure(figsize=(12, 6))
column1_count.plot(kind='bar')
plt.title('Count of Path')
plt.xlabel('Path')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

plt.tight_layout()
plt.show()


# In[14]:


# Count the values in the 'Interest' column
column2_count = df['Interest'].value_counts()

# Plot the count of 'Interest'
plt.subplot(1, 2, 2)
column2_count.plot(kind='bar')
plt.title('Count of Interest')
plt.xlabel('Interest')
plt.ylabel('Count')

plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()

# Adjust the figure size for better visibility
plt.figure(figsize=(10, 5))

plt.show()


# In[15]:


plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference

plt.hist(df['Traffic Source'], bins=20)  # Increase the number of bins for better clarity

plt.title('Count of Traffic Source')
plt.xlabel('Traffic Source')
plt.ylabel('Count')

plt.xticks(rotation=45)  # Rotate x-axis labels if necessary for better readability

plt.tight_layout()  # Adjust the spacing of the plot elements

plt.show()


# In[16]:


# Count the values in the 'Country' column
country_counts = df['Country'].value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(country_counts, labels=country_counts.index, autopct='%1.1f%%')
#plt.title('Distribution of Countries')

plt.axis('equal')  # Ensure the pie chart is circular
plt.show()


# In[18]:


import dash
#import dash_core_components as dcc
#import dash_html_components as html
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

data = pd.read_csv("olympicsdata.csv")

# Creating the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True,external_stylesheets=[dbc.themes.BOOTSTRAP])

navbar_style = {'background-color': 'lightblue', 'padding': '5px'}
link_style = {'color': 'white', 'margin-right': '10px'}

# Navbar style
navbar_style = {'background-color': 'lightblue', 'padding': '5px'}
link_style = {'color': 'white', 'margin-right': '10px'}

total_requests = data['IP Address'].count()
total_visits = data['Visits'].sum()
unique_countries = data['Country'].nunique()
country_counts = data['Country'].value_counts()

age_bin_options = [{'label': 'All', 'value': 'All'}] + [{'label': age_bin, 'value': age_bin} for age_bin in data['Age'].unique()]
country_options = [{'label': 'All', 'value': 'All'}] + [{'label': Country, 'value': Country} for Country in data['Country'].unique()]
gender_options = [{'label': 'All', 'value': 'All'}] + [{'label': Gender, 'value': Gender} for Gender in data['Gender'].unique()]
gender_counts = data['Gender'].value_counts()
age_counts = data['Age'].value_counts().sort_index()
sorted_age_groups = sorted(age_counts.index)
sorted_age_counts = [age_counts[age_group] for age_group in sorted_age_groups]
traffic_counts = data['Traffic Source'].value_counts()

fig_bar = px.bar(data, x='Age', y='Visits', color='Gender', title='Visits by Age and Gender')

# Define the scatter plot
#scatter_fig = px.scatter(data, x='Path', y='Duration', title='Path by Duration')
path_duration_sum = data.groupby('Path')['Duration'].sum().reset_index()
scatter_fig = px.bar(path_duration_sum, x='Path', y='Duration', title='Sum of Time for Each Path')

# Define the bar plot
bar_fig = px.bar(data, x='Age', y='Visits', color='Country', title='Bar Plot')

# Define the histogram
hist_fig = px.histogram(data, x='Age', y='Visits', title='Histogram')

# Define the pie chart
fig_pie_traffic = px.pie(
    values=traffic_counts.values,
    names=traffic_counts.index,
    title='Traffic Sources'
)

fig_pie_age = px.pie(
    values=sorted_age_counts,
    names=sorted_age_groups,
    title='Age Count'
)

# Define the pie chart for gender distribution
fig_pie_gender = px.pie(
    values=gender_counts.values,
    names=gender_counts.index,
    title='Gender Count'
)

fig_map = go.Figure(data=go.Choropleth(
    locations=data['Country'],
    locationmode='country names',
    z=country_counts,
    colorscale='Blues',
    colorbar_title='Count',
))
fig_map.update_layout(title='Map')

# Create the cards
card_total_requests = dbc.Card(
    dbc.CardBody([
        html.H3('Total Requests'),
        html.H4(f'{total_requests}')
    ]),
    className='card'
)

card_total_visits = dbc.Card(
    dbc.CardBody([
        html.H3('Total Visits'),
        html.H4(f'{total_visits}')
    ]),
    className='card'
)

# Count the occurrences of each user agent
# Count the occurrences of each user agent
user_agent_counts = data['User Agent'].value_counts()

# Create a DataFrame with all user agents and their counts
user_agents_table = pd.DataFrame({'User Agent': user_agent_counts.index, 'Count': user_agent_counts.values})

# Define the layout for the main page
app.layout = html.Div([
    html.Nav([
        html.H1('Fun Olympics'),
        html.A('User Demographics', href='/', style=link_style),
        html.A('Users Traffic', href='/page1', style=link_style),
        html.A('User Agents', href='/page2', style=link_style),
    ], style=navbar_style),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', className='container'),
])

# Define the layout for the home page
home_layout = html.Div([
    html.H1('User Demographics'),

    dbc.Row([
        dbc.Col(card_total_requests, width=6),
        dbc.Col(card_total_visits, width=6),
        dbc.Col(dcc.Graph(figure=fig_map), width=6),
        dbc.Col(dcc.Graph(figure=fig_pie_age), width=6),
        dbc.Col(dcc.Graph(figure=fig_pie_gender), width=6),
        dbc.Col(dcc.Graph(figure=fig_bar), width=6),

    ])
])

# Define the layout for page 1
page1_layout = html.Div([
    html.H1('Users Traffic'),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='age-dropdown',
                options=age_bin_options,
                value='All',
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='country-dropdown',
                options=country_options,
                value='All',
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='gender-dropdown',
                options=gender_options,
                value='All',
                clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block'})
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='pie-traffic', figure=fig_pie_traffic)
        ], width=6),
        dbc.Col([
            dcc.Graph(id='bar-path')
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='bar-interests')
        ], width=6),
        dbc.Col([
            dcc.Graph(id='scatter-plot', figure=scatter_fig)
        ], width=6)
    ])
], style={'height': '100vh', 'overflow-y': 'scroll'})


page2_layout = html.Div([
    html.H1('User Agents'),
    dbc.Row([
        #dbc.Col(dcc.Graph(figure=fig_map), width=6),
        #dbc.Col(dcc.Graph(figure=fig_bar), width=6)
        dbc.Table.from_dataframe(user_agents_table, striped=True, bordered=True, hover=True)

    ])
])

# Callback to update the pie chart for traffic sources
@app.callback(
    Output('pie-traffic', 'figure'),
    Input('age-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('gender-dropdown', 'value')
)
def update_pie_traffic(age, country, gender):
    filtered_data = data
    if age != 'All':
        filtered_data = filtered_data[filtered_data['Age'] == age]
    if country != 'All':
        filtered_data = filtered_data[filtered_data['Country'] == country]
    if gender != 'All':
        filtered_data = filtered_data[filtered_data['Gender'] == gender]

    traffic_counts = filtered_data['Traffic Source'].value_counts()
    fig = px.pie(
        values=traffic_counts.values,
        names=traffic_counts.index,
        title='Traffic Sources'
    )
    return fig


# Callback to update the bar chart for path and visits
@app.callback(
    Output('bar-path', 'figure'),
    Input('age-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('gender-dropdown', 'value')
)
def update_bar_path(age, country, gender):
    filtered_data = data
    if age != 'All':
        filtered_data = filtered_data[filtered_data['Age'] == age]
    if country != 'All':
        filtered_data = filtered_data[filtered_data['Country'] == country]
    if gender != 'All':
        filtered_data = filtered_data[filtered_data['Gender'] == gender]

    fig = px.bar(
        filtered_data,
        x='Path',
        y='Visits',
        title='Path by Visits'
    )
    return fig


# Callback to update the bar chart for interests and visits
@app.callback(
    Output('bar-interests', 'figure'),
    Input('age-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('gender-dropdown', 'value')
)
def update_bar_interests(age, country, gender):
    filtered_data = data
    if age != 'All':
        filtered_data = filtered_data[filtered_data['Age'] == age]
    if country != 'All':
        filtered_data = filtered_data[filtered_data['Country'] == country]
    if gender != 'All':
        filtered_data = filtered_data[filtered_data['Gender'] == gender]

    fig = px.bar(
        filtered_data,
        x='Interest',
        y='Visits',
        color='Country',
        title='Interests by Visits'
    )
    return fig


# Callback to update the scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('age-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('gender-dropdown', 'value')
)
def update_scatter_plot(age, country, gender):
    filtered_data = data
    if age != 'All':
        filtered_data = filtered_data[filtered_data['Age'] == age]
    if country != 'All':
        filtered_data = filtered_data[filtered_data['Country'] == country]
    if gender != 'All':
        filtered_data = filtered_data[filtered_data['Gender'] == gender]

    fig = px.scatter(
        filtered_data,
        x='Path',
        y='Duration',
        title='Path by Duration'
    )
    return fig


# Callback to update the page content based on the selected page
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def render_page_content(pathname):
    if pathname == '/':
        return home_layout
    elif pathname == '/page1':
        return page1_layout
    elif pathname == '/page2':
        return page2_layout
    else:
        return html.Div([
            html.H2('404 Page Not Found')
        ])


if __name__ == '__main__':
    app.run_server(debug=True)

#Note: I have removed the `fig_bar` and `fig_pie_age` from the home layout as they were not being used. Feel free to add them back if needed. Also, make sure the file "olympicsdata.csv" exists in the same directory as the Python script.


# In[ ]:




