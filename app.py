# Imports
import pandas as pd
import numpy as np

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import random
import dash
import json
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_daq as daq


# ------------------------------------- Our Model-------------------------------------#
# data import
noon_clean = pd.read_csv('./data/noon_clean_Model.csv')
# filling NA's with empty string
noon_clean=noon_clean.fillna('')

# --------------------------------- Generate Popularity Column--------------------------#
import random
for i in noon_clean.index:
    if (noon_clean.at[i, 'Scents/Notes']=="Floral"):
        noon_clean.at[i, 'Popularity'] = random.randrange(1650, 1800, 10)
        
    elif(noon_clean.at[i, 'Scents/Notes']=="Woody"):
        noon_clean.at[i, 'Popularity'] = random.randrange(1550, 1700, 10)  
        
    elif(noon_clean.at[i, 'Scents/Notes']=="Fresh"):
        noon_clean.at[i, 'Popularity'] = random.randrange(1450, 1600, 10)
        
    elif(noon_clean.at[i, 'Scents/Notes']=="Oriental"):
        noon_clean.at[i, 'Popularity'] = random.randrange(1350, 1500, 10)
        
    elif(noon_clean.at[i, 'Scents/Notes']=="Fruity"):
        noon_clean.at[i, 'Popularity'] = random.randrange(1250, 1400, 10) 
        
    elif(noon_clean.at[i, 'Scents/Notes']=="Arabian"):
        noon_clean.at[i, 'Popularity'] = random.randrange(1150, 1300, 10) 
        
    elif(noon_clean.at[i, 'Scents/Notes']=="Spicy"):
        noon_clean.at[i, 'Popularity'] = random.randrange(1050, 1200, 10)
        
    elif(noon_clean.at[i, 'Scents/Notes']=="Citrus"):
        noon_clean.at[i, 'Popularity'] = random.randrange(950, 1100, 10) 
        
    elif(noon_clean.at[i, 'Scents/Notes']=="Aromatic"):
        noon_clean.at[i, 'Popularity'] = random.randrange(850, 1000, 10)
        
    elif(noon_clean.at[i, 'Scents/Notes']=="Vanilla"):
        noon_clean.at[i, 'Popularity'] = random.randrange(750, 900, 10)
        
    elif(noon_clean.at[i, 'Scents/Notes']=="Musk"):
        noon_clean.at[i, 'Popularity'] = random.randrange(650, 800, 10)
        
    elif(noon_clean.at[i, 'Scents/Notes']=="Sweet"):
        noon_clean.at[i, 'Popularity'] = random.randrange(550, 700, 10)
        
    elif(noon_clean.at[i, 'Scents/Notes']=="Jasmine"):
        noon_clean.at[i, 'Popularity'] = random.randrange(450, 600, 10)     
        
    elif(noon_clean.at[i, 'Scents/Notes']=="Sandalwood"):
        noon_clean.at[i, 'Popularity'] = random.randrange(350, 500, 10)
        
    elif(noon_clean.at[i, 'Scents/Notes']=="Clean"):
        noon_clean.at[i, 'Popularity'] = random.randrange(250, 400, 10)   
noon_clean.Popularity=noon_clean.Popularity.astype(str)


# get copy to work with cos sim and binary matrix 
noon_bin = noon_clean.copy()
# applying the split by ',' to the base note column
noon_bin['Base Note']=noon_bin['Base Note'].apply(lambda x: x.split(","))
noon_bin['Heart/Middle Note']=noon_bin['Heart/Middle Note'].apply(lambda x: x.split(","))
noon_bin['Top Note']=noon_bin['Top Note'].apply(lambda x: x.split(","))


from collections import Counter

# Top note freq counter
top_note_counts = Counter(tn for top_note in noon_bin['Top Note'] for tn in top_note)

# middle note freq counter
middle_note_counts = Counter(mn for middle_note in noon_bin['Heart/Middle Note'] for mn in middle_note)

# base note freq counter
base_note_counts = Counter(bn for base_note in noon_bin['Base Note'] for bn in base_note)

# popularity freq counter
pop_counter=Counter(noon_bin[noon_bin['Popularity'].notnull()]['Popularity'])



from itertools import dropwhile
# drop any note that has not been repeated at least once, because for our recommendation system to work we need at least 2
# frequencies of a note to make the recommendation

# top notes
for key, count in dropwhile(lambda key_count: key_count[1] >= 2, top_note_counts.most_common()):
    del top_note_counts[key]

# middle notes
for key, count in dropwhile(lambda key_count: key_count[1] >= 2, middle_note_counts.most_common()):
    del middle_note_counts[key]
    
# base notes    
for key, count in dropwhile(lambda key_count: key_count[1] >= 2, base_note_counts.most_common()):
    del base_note_counts[key]


# this is to handle the null values that were replaced by an empty string
del top_note_counts['']
del middle_note_counts['']
del base_note_counts['']


# we need to create a binary matrix for the selected features, where the rows represent each perfume and the columns 
# represent the features, 1 means this perfume has this note and 0 means it does not

#-------------------------------------- TOP NOTE--------------------------------------#

top_notes = list(top_note_counts.keys())

# create the binary matrix for top notes
for tn in top_notes:
    noon_bin[tn] = noon_bin['Top Note'].transform(lambda x: int(tn in x))
    
#-------------------------------------- MIDDLE NOTE-----------------------------------#

middle_notes = list(middle_note_counts.keys())

# create the binary matrix for middle notes
for mn in middle_notes:
    noon_bin[mn] = noon_bin['Heart/Middle Note'].transform(lambda x: int(mn in x))
#-------------------------------------- BASE NOTE-------------------------------------#

base_notes = list(base_note_counts.keys())

# create the binary matrix for base notes
for bn in base_notes:
    noon_bin[bn] = noon_bin['Base Note'].transform(lambda x: int(bn in x))
    
#--------------------------------------POPULARITY-------------------------------------#
popularity_keys=list(pop_counter.keys())

# create the binary matrix for popularity
for pop in popularity_keys:
    noon_bin[pop] = noon_bin['Popularity'].transform(lambda x: int(pop in x))


# save the binary matrix in a new variable

# top note
perfume_features_TN = noon_bin[top_notes]

# middle note
perfume_features_MN = noon_bin[middle_notes]

# base note
perfume_features_BN = noon_bin[base_notes]

# popularity
popularity_features_BN = noon_bin[popularity_keys]

# concat all the notes to get all features
perfume_features_pop = pd.concat([perfume_features_TN,perfume_features_MN,perfume_features_BN,popularity_features_BN], axis=1)
perfume_features=pd.concat([perfume_features_TN,perfume_features_MN,perfume_features_BN], axis=1)


from sklearn.metrics.pairwise import cosine_similarity

# create a cosine similarity matrix using the binary matrix
# all 
cosine_sim_All = cosine_similarity(perfume_features, perfume_features)

# All note+popularity
cosine_sim_All_pop = cosine_similarity(perfume_features_pop, perfume_features_pop)

# top note
cosine_sim_TN = cosine_similarity(perfume_features_TN, perfume_features_TN)

# middle note
cosine_sim_MN = cosine_similarity(perfume_features_MN, perfume_features_MN)

# base note
cosine_sim_BN = cosine_similarity(perfume_features_BN, perfume_features_BN)

from fuzzywuzzy import process

# using the fuzzy wuzzy package, we can get the exact name of the perfume even if its misspelled 
def perfume_finder(name):
    all_names = noon_clean['name'].tolist()
    closest_match = process.extractOne(name,all_names)
    return closest_match[0]

# we need to get the index of the 'Azurl' perfume in the cosine sim matrix to find recommendations for it

# this is a dictionary where the keys are perfume names and the values are perfume indices
perfume_idx = dict(zip(noon_bin['name'], list(noon_bin.index)))

# this function takes the name of the perfume and how many recommendations and returns them both
def get_content_based_recommendations(name_string, note_type, rec_type='match',depart_filter='off' ):
    name = perfume_finder(name_string)
    idx = perfume_idx[name]
    rec_list=[]
    better_rec=[]  
    selected_depart=noon_clean[noon_clean['name']==name]['Department'].values[0]
    n_recommendations=30

    # for base note 
    if note_type == 'base':
        
        sim_scores = list(enumerate(cosine_sim_BN[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:(n_recommendations+1)]
        similar_perfumes = [i[0] for i in sim_scores]      
        list_names=list(noon_bin['name'].iloc[similar_perfumes])
        for i in list_names: 
            rec_depart=noon_clean[noon_clean['name']==i]['Department'].values[0]
            if(rec_depart==selected_depart):
                better_rec.append(i)
                
        rec_list=list(noon_bin['name'].iloc[similar_perfumes])

    # for top note
    elif note_type == 'top': 
        
        sim_scores_t = list(enumerate(cosine_sim_TN[idx]))
        sim_scores_t = sorted(sim_scores_t, key=lambda x: x[1], reverse=True)
        sim_scores_t = sim_scores_t[1:(n_recommendations+1)]
        similar_perfumes_t = [i[0] for i in sim_scores_t]       
        list_names=list(noon_bin['name'].iloc[similar_perfumes_t])
        for i in list_names: 
            rec_depart=noon_clean[noon_clean['name']==i]['Department'].values[0]
            if(rec_depart==selected_depart):
                better_rec.append(i)
        
        rec_list=list(noon_bin['name'].iloc[similar_perfumes_t])
        
    # for middle note    
    elif note_type == 'middle': 
        
        sim_scores_m = list(enumerate(cosine_sim_MN[idx]))
        sim_scores_m = sorted(sim_scores_m, key=lambda x: x[1], reverse=True)
        sim_scores_m = sim_scores_m[1:(n_recommendations+1)]
        similar_perfumes_m = [i[0] for i in sim_scores_m]
        list_names=list(noon_bin['name'].iloc[similar_perfumes_m])
        for i in list_names: 
            rec_depart=noon_clean[noon_clean['name']==i]['Department'].values[0]
            if(rec_depart==selected_depart):
                better_rec.append(i)
        rec_list=list(noon_bin['name'].iloc[similar_perfumes_m])
        
    # for all note    
    elif note_type == 'all_pop': 
        
        sim_scores_a = list(enumerate(cosine_sim_All_pop[idx]))
        sim_scores_a = sorted(sim_scores_a, key=lambda x: x[1], reverse=True)
        sim_scores_a = sim_scores_a[1:(n_recommendations+1)]
        similar_perfumes_a = [i[0] for i in sim_scores_a]
        list_names=list(noon_bin['name'].iloc[similar_perfumes_a])
        for i in list_names: 
            rec_depart=noon_clean[noon_clean['name']==i]['Department'].values[0]
            if(rec_depart==selected_depart):
                better_rec.append(i)
        rec_list=list(noon_bin['name'].iloc[similar_perfumes_a])
        
    # for all note    
    elif note_type == 'all': 
        
        sim_scores_a2 = list(enumerate(cosine_sim_All[idx]))
        sim_scores_a2 = sorted(sim_scores_a2, key=lambda x: x[1], reverse=True)
        sim_scores_a2 = sim_scores_a2[1:(n_recommendations+1)]
        similar_perfumes_a2 = [i[0] for i in sim_scores_a2]
        list_names=list(noon_bin['name'].iloc[similar_perfumes_a2])
        for i in list_names: 
            rec_depart=noon_clean[noon_clean['name']==i]['Department'].values[0]
            if(rec_depart==selected_depart):
                better_rec.append(i)        
        rec_list=list(noon_bin['name'].iloc[similar_perfumes_a2])
        
    if(depart_filter=='on'):
        if(rec_type=='match'):
            better_rec=better_rec[0]
        elif(rec_type=='complement'):
            better_rec=better_rec[0:3]
    elif(depart_filter=='off'):
        if(rec_type=='match'):
            better_rec=rec_list[0]
        elif(rec_type=='complement'):
            better_rec=rec_list[0:3]
            
    return better_rec

# ----------------------------------------------------------------------------------#

# external bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],suppress_callback_exceptions = True)
#server=app.server


# sidebar style 

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


#---------------------------------------------- sidebar content -------------------------------------------------#
sidebar = html.Div(
    [
        html.Img(src='https://static.wikia.nocookie.net/lego/images/2/23/PPG_logo.png', 
                 width="220"),
        #html.H3("Plotly Dash", className="display-4"),
        html.Hr(),
        html.P(
            "Power Puff Girls", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Try It", href="/", active="exact"),
                dbc.NavLink("About Us", href="/page-1", active="exact"),
                
                
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([
    html.Div([dcc.Location(id="url"), sidebar, content]),
    
])

#---------------------------------------------- Page content ----------------------------------------------------#

@app.callback(Output("page-content", "children"), 
              [Input("url", "pathname")])

def render_page_content(pathname):
    
    # Create labebls for pie chart input
    department_labels = ['Unisex','Male','Female']
    perfume_names = list(noon_clean['name'])
    recomm_type = ['all', 'top' , 'base', 'middle']
    
    # contact us table 
    table_header = [    html.Thead(html.Tr([html.Th("Member"), html.Th("Email"), html.Th("GitHub")]))]
    row1 = html.Tr([html.Td(["Sara Aldubaie"]), html.Td("sara.aldubaie@gmail.com"), html.Td(html.A("GitHub Account", href='https://github.com/Sara-Aldubaie', target="_blank"))])
    row2 = html.Tr([html.Td(["Wejdan Al-Ahmadi"]), html.Td("wejdan.alahmadi94@gmail.com"), html.Td(html.A("GitHub Account", href='https://github.com/wejdani', target="_blank"))])
    table_body = [html.Tbody([row1, row2])]
    #--------------------------------------------- About Us page--------------------------------------------------# 
    if pathname == "/page-1":
        
        q_style = {'font-weight': 'bold'}
        return  (html.Div([
                html.Center(html.H1('About Us')),
            
                #html.P(["Our team?"], style = q_style) ,  Add later  
            
                html.P(["What we made? "], style = q_style),
                html.P(["We made a recommender system that shows you not only a perfume that matches your entered choice, it can also suggest three different perfumes that are compatible with it."]),
            
                html.P(["Where to find us?"], style = q_style),
                # Email follow me on Github
                dbc.Table(table_header + table_body, bordered=True),
                #html.P([""])
                
                ], className="pretty_container", 
            style = { 'font-family': '"Times New Roman", Times, serif', "width":"70%"})
    
                )
    
    #-------------------------------------------------- Try it page----------------------------------------------# 
    elif pathname == "/":
        return  (html.Div([
                # Title
                html.Center(html.H1("Noon Perfume Recommender")),
            
            
                    # First Dynamic Input(BooleanSwitch)
                    html.Label(["Filter by Department:",]),
                    html.Br([]),
                    html.Label([
                    daq.BooleanSwitch(
                      id='booleanswitch',
                      on=False,
                          label="",
                      labelPosition="top"
                    )]),
            
                    # Line Break
                    html.Br([]),
                    html.Br([]),
            
                    # Second Dynamic Input(RadioItems)
                    html.Label(["Please Select The Type of Recommendation:",
                    dcc.RadioItems(
                    id="recomm_rad",
                    options=[{'label':'All Notes', "value":'all'},
                             {'label':'All Notes & Popularity', "value":'all_pop'},
                             {'label':'Top Notes', "value":'top'},
                             {'label':'Middle Notes', "value":'middle'},
                             {'label':'Base Notes', "value":'base'}
                            ],
                    value='all',
                                #styling
                    labelStyle = {'display': 'inline', 'cursor': 'pointer', 'margin':'10px'},
                    inputStyle={"margin-right": "10px"}),
                    #labelStyle={'display': 'inline-block'}) 
                    ]),

                    # Line Break
                    html.Br([]),
                    
                    html.Br([]),
            

                    # Third Dynamic Input(RadioItems)
            
                    html.Label(["Please Select The Type of Perfume:",
                                dcc.RadioItems(
                                id="department",
                                options=[{'label':'Unisex', "value":'Unisex'},
                                        {'label':'Women', "value":'Women'},
                                        {'label':'Men', "value":'Men'}],
                                value='Unisex',
                                #styling
                                labelStyle = {'display': 'inline', 'cursor': 'pointer', 'margin':'10px'},
                                inputStyle={"margin-right": "10px"}),
                                #labelStyle={'display': 'inline-block'}) 
                               ]),

                    # Line Break
                    html.Br([]),
                    
                    html.Br([]),

                    # Forth Dynamic (Dropdown)
        
                    html.Label(["Choose Perfume: ",
                                dcc.Dropdown(
                                id='perf_names', 
                                clearable=False,
                                value= "Azur", 
                                options=[{'label':x, "value":x} for x in perfume_names])], style={"width": "100%"},),


                    # Line Break
                    html.Br([]),
                    
                    html.Br([]),
          html.Div([
                    # Fifth Dynamic Input (Button)
                    dbc.Button("Find Match",
                               id ='btn-nclicks-1' ,
                               size="lg", 
                               outline=True, 
                               color="primary", 
                               block=True,
                               className="mr-1",
                              n_clicks=0),
            
                    # Sixth Dynamic Input (Button)
                    dbc.Button("Complement",
                               id ='btn-nclicks-2' ,
                               size="lg", 
                               outline=True, 
                               color="primary", 
                               block=True,
                               className="mr-1",
                              n_clicks=0),
              html.Br([]),
              html.Br([]),
          html.Div(id='container-button-timestamp')  ]),

                    # Line Break
                    html.Br([]),

                    html.Div(id='body-div')


                        ], className="pretty_container", 
            style = {'text-transform': 'capitalize', 'font-family': '"Times New Roman", Times, serif', "width":"70%"})
                    )
    
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )



#-------------------------------------------------------------------------------------------------------#



# -------------------------------------- For changing the name drop down list based on radio button --------------------#
@app.callback(
    [Output('perf_names', 'options'),
     Output('perf_names', 'value')],
    Input('department', 'value'))

def dropdown_options(radio_value):
    # retrieve perfume names based on radio button input
    perfume_names = noon_clean[noon_clean['Department']==radio_value]['name']
    
    # convert to list to get indexes
    perfume_names=list(perfume_names)
    
    # populate the options and values of the drop down using the retrieved list of names
    options = [{'label': x, 'value': x} for x in list(perfume_names)]
    value = perfume_names[0]
    
    return options, value


# -------------------------------------- For getting perfume recommendation from drop down --------------------#
        
@app.callback(Output('container-button-timestamp', 'children'),
              Input('btn-nclicks-1', 'n_clicks'),
              Input('btn-nclicks-2', 'n_clicks')
             ,Input('recomm_rad','value'),
              Input('booleanswitch','on'),
     state=[State(component_id='perf_names', component_property='value')])
def displayClick(btn1, btn2,recomm_rad,switch,perf_names):
    toggle = 'off'
    if(switch==True):
        toggle='on'
    if(switch==False):
        toggle='off'
            
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    
    
    # find Match
    if 'btn-nclicks-1' in changed_id:
        rec_perfs = get_content_based_recommendations(perf_names,recomm_rad,'match',toggle) 
        msg= f'recommended perfume for {perf_names} is:{rec_perfs}'
        name=rec_perfs
        perfume=noon_clean[noon_clean['name']==name][['name','Department','price','link']]
        price = str(perfume['price'].values[0])+' SR'
        return html.Div(
            
                       [html.Table([
            html.Tr([html.Th(['Perfume:'],style = {'background-color': '#dcedf5'}),
                    html.Td(perfume['name'].values[0] )]),
            html.Tr([html.Th(['Price:'],style = {'background-color': '#dcedf5'}),
                     html.Td(price)]),
            html.Tr([html.Th(['Type:'],style = {'background-color': '#dcedf5'}),
                     html.Td(perfume['Department'].values[0]   )]),
            html.Tr([html.Th(''),
                     html.Td(html.A(dbc.Button('Buy me!',color="primary",className="mr-1",), 
                                    href=perfume['link'].values[0],target='_blank'))])
        ],
                style={'table-layout': 'fixed','marginLeft': 'auto', 'marginRight': 'auto','border': 'solid',
                  'border-width': '0.2px', 'minWidth': '30%', 'width': '30%', 'maxWidth': '30%', 
                      'border-spacing': '5px','border-collapse': 'separate'},
                       ), 
                       
                       ])
    
    # find Complement

    elif 'btn-nclicks-2' in changed_id:
        rec_perfs = get_content_based_recommendations(perf_names,recomm_rad,'complement',toggle)
        msg= f'recommended perfume for {perf_names} is:{rec_perfs[0]}, and {rec_perfs[1]}, and {rec_perfs[2]}'
        
        name1=rec_perfs[0]
        perfume1=noon_clean[noon_clean['name']==name1][['name','Department','price','link']]
        
        price1 = str(perfume1['price'].values[0])+' SR'
        
        name2=rec_perfs[1]
        perfume2=noon_clean[noon_clean['name']==name2][['name','Department','price','link']]
        price2 = str(perfume2['price'].values[0])+' SR'
        
        name3=rec_perfs[2]
        perfume3=noon_clean[noon_clean['name']==name3][['name','Department','price','link']]
        price3 = str(perfume2['price'].values[0])+' SR'
        
        
        return html.Div(
            html.Table([
            html.Tr([html.Th(['Perfume:'],style = {'background-color': '#dcedf5'}),
                     html.Td(perfume1['name'].values[0]),
                    html.Th(['Perfume:'],style = {'background-color': '#dcedf5'}),
                     html.Td([perfume2['name'].values[0]]),
                    html.Th(['Perfume:'],style = {'background-color': '#dcedf5'}),
                   html.Td(perfume3['name'].values[0]),
                    ] ),
            html.Tr([html.Th(['Price:'],style = {'background-color': '#dcedf5'}),
                      html.Td(price1),
                     html.Th(['Price:'],style = {'background-color': '#dcedf5'}),
                     html.Td([price2]),
                    html.Th(['Price:'],style = {'background-color': '#dcedf5'}),
                    html.Td(price3)
                    ],),
            html.Tr([html.Th(['Type:'],style = {'background-color': '#dcedf5'}),
                     html.Td(perfume1['Department'].values[0]),
                     html.Th(['Type:'],style = {'background-color': '#dcedf5'}),
                     html.Td([perfume2['Department'].values[0]]),
                    html.Th(['Type:'],style = {'background-color': '#dcedf5'}),
                    html.Td(perfume3['Department'].values[0]),
                    ]),
            html.Tr([html.Th(''),

                     html.Td(html.A(dbc.Button('Buy me!',color="primary",className="mr-1"), 
                            href=perfume1['link'].values[0],target='_blank')),
                     html.Th(''),
                    html.Td(html.A(dbc.Button('Buy me!',color="primary",className="mr-1"), 
                            href=perfume2['link'].values[0],target='_blank')),
                     html.Th(''),
                    html.Td(html.A(dbc.Button('Buy me!',color="primary",className="mr-1"), 
                            href=perfume3['link'].values[0],target='_blank'))
                    ])
          
        ],style={'table-layout': 'fixed','auto': '300px', 'marginRight': 'auto','border': 'solid',
              'border-width': '0.2px','minWidth': '100%', 'width': '100%', 'maxWidth': '100%',
                 'padding': '15px','cellpadding-left':'10px', 'border-spacing': '3px','border-collapse': 'separate'  })

        )

    else:
        msg = ''
   # return html.Div(msg)


# Note: if you have more than one input or output make sure to have them in a list

if __name__ == '__main__':
    app.run_server(port=8054,debug=False) # or whatever you choose
