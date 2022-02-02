import streamlit as st
import category_encoders as ce
import pandas as pd
import numpy as np
import plotly.express as px
#streamlit supports most major plotting libraries - cool!
import pickle


st.title("Car Pollution over Time")

url = r'https://raw.githubusercontent.com/ethlala/gazelle-4K/main/ELA_final_project/fuel_efficiency.csv'


num_rows = st.sidebar.number_input('Select # of Rows to Load (dataset is 38,113 rows', 
                                    min_value = 10000, 
                                    max_value = 38113, 
                                    step = 5000)

section = st.sidebar.radio('Choose Application Section', ['Data Explorer', 'Model Explorer'])

#cache this expensive step so it doesn't slow us down every time
#decorators are commands you can call that influences the function below it 
@st.cache #this is a decorator
def load_data(num_rows):
    df = pd.read_csv(url,nrows = num_rows)
    df['pollution'] = [1 if x <= 390 else 3 if x >= 560 else 2 for x in df['Tailpipe CO2 in Grams/Mile (FT1)']]
    return df


#creating another function w/o a ton of logic so we can cache it 
#@st.cache
def create_grouping(x_axis, y_axis):
    grouping = df.groupby(x_axis)[y_axis].mean()
    return grouping

def load_model():
    with open('pipe3.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

df= load_data(num_rows)

if section == 'Data Explorer':
    #using a pandas method that returns all categorical variables
    x_axis = st.sidebar.selectbox("Choose column for x-axis", 
                                df.select_dtypes(include = np.object).columns.tolist())
    y_axis = st.sidebar.selectbox("Choose column for y-axis", 
                                ['Combined MPG (FT1)', 'Tailpipe CO2 in Grams/Mile (FT1)'])

    chart_type = st.sidebar.selectbox("Choose a chart Type", ['histogram','line', 'bar', 'area', 'trend'])

    #do a timeline chart 
    if chart_type == 'histogram':
        st.subheader('Number of rides by hour')
        fig = px.histogram(df, x="hour", y = 'count', nbins=24)
        st.plotly_chart(fig)
    elif chart_type == 'line':
        grouping = create_grouping(x_axis, y_axis)
        st.line_chart(grouping)
    elif chart_type == 'bar':
        grouping = create_grouping(x_axis, y_axis)
        st.bar_chart(grouping)
    elif chart_type == 'area':
        fig = px.strip(df[[x_axis, y_axis]], x = x_axis, y = y_axis)
        st.plotly_chart(fig)
    elif chart_type == 'trend':
        data = df.groupby(['year','month'])['count'].sum().reset_index()
        fig = px.line(data, x='month', y='count', facet_row = 'year', title='Monthly Rides Over Time')
        #fig = px.strip(df[[x_axis, y_axis]], x = x_axis, y = y_axis)
        st.plotly_chart(fig)
    st.write(df)
else:
    st.text("Let's see how big of a polluter your car is")
    model = load_model()

    class_choice = st.sidebar.selectbox("What class of car?", 
                                df['Class'].unique().tolist())
    makes = df['Make'].loc[df['Class']== class_choice]
    make_choice = st.sidebar.selectbox('Select your vehicle:', makes)
    models = df['Model'].loc[df['Make']==make_choice]
    model_choice = st.sidebar.selectbox('Select your vehicle:', models)
    years = df["Year"].loc[df['Model'] = model_choice]
    year_choice = st.sidebar.selectbox('Select year', years) 

    sample = {
    'Class': class_choice,
    'Year': year_choice,
    'Model': model_choice,
    'Make': make_choice,
    # I think I need to drop vehicle ID from the model to make this work
    # 'Vehicle ID': df['Vehicle_ID'] == 
    }

    sample = pd.DataFrame(sample, index=[0])
    st.write(sample)
    prediction = model.predict(sample)[0]

    st.title(f"Predicted Rides tomorrow at {hour}:00: {int(prediction)}")
    #surfacing the live prediction!!!
print(num_rows)
print(section)

##Notes
#Streamlit re-runs all code anytime something is changed
#This is why caching with decorators becomes important 
#