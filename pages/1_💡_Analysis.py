import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

col1, col2 = st.columns([0.1,2])
with col2:
    st.title("Flight Customer Data Analysis")

st.write("--------------------------------------------------------------------------------------------")

#Buka File
# with open("data/16-flight-EDA.pickle",'rb') as f:
#     data = pickle.load(f)
data = pd.read_csv("./data/16-flight-EDA.csv")
data['FIRST_FLIGHT_DATE_2'] = pd.to_datetime(data['FIRST_FLIGHT_DATE_2'], utc=False)

#Input tanggal
min_date = data['FIRST_FLIGHT_DATE_2'].min()
max_date = data['FIRST_FLIGHT_DATE_2'].max()
start_date, end_date = st.sidebar.date_input(label='Tanggal Penerbangan Pertama',
                                            min_value=min_date,
                                            max_value=max_date,
                                            value=[min_date, max_date])

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 300px;
        margin-left: -400px;
    }
     
    """,
    unsafe_allow_html=True,
)


st.sidebar.title("About")
st.sidebar.info(
        """
        This web [app](.....) is maintained by Nurul Fitri F
    """
    )


# img = Image.open('data/GitHub-Mark.png')
# st.sidebar.image(img)


# # Input kategori
# prov = ["All Province"] + list(data['WORK_PROVINCE'].value_counts().keys().sort_values())
# st.sidebar.selectbox(label='Provinsi', options=prov)

# if prov != "All Province":
#     list_city = data[data['WORK_PROVINCE'] == prov]
#     city = ["All City"] + list(list_city.value_counts().keys().sort_values())
#     # list_city = data[data['WORK_PROVINCE'] == prov]
# elif prov == "All Province":
#     list_city = data

# city = ["All City"] + list(data['WORK_CITY'].value_counts().keys().sort_values())
# st.sidebar.selectbox(label='City', options=city)

# Filter 
outputs = data[(data['FIRST_FLIGHT_DATE_2'] >= start_date) &
                (data['FIRST_FLIGHT_DATE_2'] <= end_date)]

# if prov != "All Province":
#     outputs = outputs[(outputs['WORK_PROVINCE'] == prov)]

# if city != "All Province":
#     outputs = outputs[(outputs['WORK_CITY'] == city)]




st.header('Table Preview')
st.checkbox("Use container width", value=False, key="use_container_width")
st.dataframe(outputs, use_container_width=st.session_state.use_container_width)


def grafik_bar(data_1, nama):
    fig = px.bar(data_1, color=data_1, orientation='v', 
          color_continuous_scale = "blues", 
          labels={"index": nama,
                  "value" : "FREQUENCY"},
          title =f'<b>TOP 10 {nama}<b>')
    st.plotly_chart(fig, theme="streamlit")

st.header('Univariate Analysis')
st.subheader('Categorical')
st.markdown('▪️ <b>Graph of the passengers origin city based on the highest frequency of passengers<b>', unsafe_allow_html=True)
col1, col2, col3,col4,col5 = st.columns([0.1,1.1,0.4,0.7,0.1])
with col2:
    bar_data = outputs['WORK_CITY'].value_counts().nlargest(10)
    kolom = 'CITY'
    grafik_bar(bar_data, kolom)
st.markdown('<div style="text-align:justify;"> Most customers come from the city of Guangzhou with 9298 customers or 16.50% of the total data. Followed by Beijing with 12,43% of the total data and Shanghai with 7.8% of the total data. </div>', unsafe_allow_html=True)

st.text("")

with col4:
    st.text("")
    bar_data = outputs['WORK_CITY'].value_counts().reset_index()
    bar_data.columns = ['CITY', 'FREQ']
    bar_data['PERCENT'] = round((bar_data['FREQ']/bar_data['FREQ'].sum())*100,3)
    st.table(bar_data.head(10))

st.markdown('▪️ <b> Graph of the passengers origin province based on the highest frequency of passengers<b>', unsafe_allow_html=True)
col1, col2, col3,col4,col5 = st.columns([0.1,1.1,0.4,0.7,0.1])
with col2:
    bar_data = outputs['WORK_PROVINCE'].value_counts().nlargest(10)
    kolom = 'PROVINCE'
    fig = grafik_bar(bar_data, kolom)
with col4:
    st.text("")
    bar_data = outputs['WORK_PROVINCE'].value_counts().reset_index()
    bar_data.columns = ['PROVINCE', 'FREQ']
    bar_data['PERCENT'] = round((bar_data['FREQ']/bar_data['FREQ'].sum())*100,3)
    st.table(bar_data.head(10))
st.markdown('<div style="text-align:justify;"> Most customers come from Guangdong Province with a customer frequncy is 17,346 customers or 30,79% of the total data. Followed by Beijing with 12,77% of the total data and Shanghai with 7.8% of the total data. </div>', unsafe_allow_html=True)
st.text("")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('▪️ <b> Graph of the passengers origin country based on the highest frequency of passengers<b>', unsafe_allow_html=True)
col1, col2, col3,col4,col5 = st.columns([0.1,1.1,0.4,0.7,0.1])
with col2:
    bar_data = outputs['WORK_COUNTRY'].value_counts().nlargest(3)
    kolom = 'COUNTRY'
    fig = grafik_bar(bar_data, kolom)
with col4:
    st.text("")
    bar_data = outputs['WORK_COUNTRY'].value_counts().reset_index()
    bar_data.columns = ['COUNTRY', 'FREQ']
    bar_data['PERCENT'] = round((bar_data['FREQ']/bar_data['FREQ'].sum())*100,3)
    st.table(bar_data.head(10))
st.markdown('<div style="text-align:justify;"> From this data it can be seen that most of the countries of origin of customers are from China by 94.5%. List of some city in China is Guangzhou, Beijing, Shanghai, Shenzhen, and Dalian.</div>', unsafe_allow_html=True)

st.text("")

st.markdown('▪️ <b>Graph of the passengers gender based on the highest frequency of passengers<b>', unsafe_allow_html=True)
col1, col2, col3,col4,col5 = st.columns([0.1,1.1,0.4,0.7,0.1])
with col2:
    bar_data = outputs['GENDER'].value_counts().nlargest(3)
    kolom = 'GENDER'
    fig = grafik_bar(bar_data, kolom)
with col4:
    st.text("")
    bar_data = outputs['GENDER'].value_counts().reset_index()
    bar_data.columns = ['GENDER', 'FREQ']
    bar_data['PERCENT'] = round((bar_data['FREQ']/bar_data['FREQ'].sum())*100,3)
    st.table(bar_data.head(10))
st.markdown('<div style="text-align:justify;"> Male customers travel more than female customers, it can be seen in the graph that 76% of male customers and only 23.6% of female customers</div>', unsafe_allow_html=True)
st.text("")

st.subheader('Numeric')

# width = st.sidebar.slider("plot width", 1, 25, 3)
# height = st.sidebar.slider("plot height", 1, 25, 1)

col1, col2, col3 = st.columns([0.5,4,1])
with col2:
    f,ax = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(14, 9)
    )

    color = ['darkmagenta','darkblue','g','red','black','pink','black','purple','green']
    features = ['AGE','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount','EXCHANGE_COUNT','Points_Sum','Point_NotFlight','AVG_INTERVAL']
    for i in range(0, len(features)):
        plt.subplot(3, len(features)//2-1, i+1)
        fg = sns.histplot(x=outputs[features[i]], color=color[i], kde=True)
        fg.set_title(features[i] + ' Distribution')
        plt.xlabel(features[i])
    plt.tight_layout()
    st.pyplot(fig)

st.markdown('<div style="text-align:justify;"> From the graph, we can see the distribution of numerical data such as the age distribution of the passengers, the distance from the last flight to the last passenger''s flight order, the number of customer flights, the total flight distances that have been carried out, the average discount the customer received, the number of points earned by customers. From the graph it can be seen that there are 7 histograms that have a positive skew scheme dan 2 histograms that have a normal distribution. <br> Conclusion based on data: <br> - The age group that makes the most flights is in the age range of 35-48 years. <br> - The last flight time interval to the most recent flight order, is in the range of 28 days to 257 days. The LAST_TO_END graph is not evenly distributed or it can be said to have a positive skew scheme. <br> - The number of flights made by customers is at most 3 to 15 flights. <br> - The frequency of the total distance (km) of flights that appears the most is in the range of 4km to 21km. <br> - The average discount received by customers is in the range of 23% - 81%.</div>', unsafe_allow_html=True)

# 11,7

st.header('Multivariate Analysis')
st.subheader('Correlation')
# width = st.slider("plot width", 1, 25, 3)
# height = st.slider("plot height", 1, 25, 1)
col1, col2, col3 = st.columns([1,4,1.5])
with col2:
    corr_= outputs[features].corr()
    plt.figure(figsize=(15,8))
    sns.heatmap(corr_, annot=True, fmt = ".2f", cmap = "BuPu")
    st.pyplot(fig)

st.text("")
st.subheader('Pair Plot Correlation')
features = ['SEG_KM_SUM','avg_discount','EXCHANGE_COUNT','Points_Sum','MEMBERSHIP','FLIGHT_COUNT']
col1, col2, col3 = st.columns([1,4,1.5])
with col2:
	# fig = sns.pairplot(outputs[features], hue="FLIGHT_COUNT") 
	# st.pyplot(fig)
    img = Image.open('data/Pairplot.png')
    st.image(img)
