import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

# col1, col2 = st.columns([0.5,2])
# with col2:
st.title("Clustering Flight Customers Using Unsupervised Learning")

st.text("------------------------------------------------------------------------")


#Buka File
with open("data/16-flight-EDA.pickle",'rb') as f:
    data = pickle.load(f)

data_sel = data[['MEMBERSHIP','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','avg_discount']]

st.header('Data Preparation')
st.subheader('Feature Selection')
col1, col2, col3 = st.columns([1,4,1.5])
with col2:
	# fig = sns.pairplot(outputs[features], hue="FLIGHT_COUNT") 
	# st.pyplot(fig)
    img = Image.open('data/Feature.png')
    st.image(img)

st.markdown('<div style="text-align:justify;">Pada pembuatan clustering flight costumer, fitur yang digunakan adalah LRFMC, yaitu the end time of observation window - the time of membership (MEMBERSHIP), Jarak waktu penerbangan terakhir ke pesanan penerbangan paling akhir (LAST_TO_END), Jumlah penerbangan costumer (FLIGHT_COUNT), Total jarak (km) penerbangan yang sudah dilakukan (SEG_KM_SUM), rata-rata discount yang didapat costumer (AVG_DISCOUNT).</div>', unsafe_allow_html=True)

st.write("")

st.subheader('Data Preview')
st.checkbox("Use container width", value=False, key="use_container_width")
st.dataframe(data_sel, use_container_width=st.session_state.use_container_width)

st.text('')

data_sel = data_sel.drop_duplicates()
data_sebelum = data_sel.copy()


#log Transformation
data_sel['MEMBERSHIP'] = np.log1p(data_sel['MEMBERSHIP'])
data_sel['LAST_TO_END'] = np.log1p(data_sel['LAST_TO_END'])
data_sel['FLIGHT_COUNT'] = np.log1p(data_sel['FLIGHT_COUNT'])
data_sel['SEG_KM_SUM'] = np.log1p(data_sel['SEG_KM_SUM'])
data_sel['avg_discount'] = np.log1p(data_sel['avg_discount'])

# Remove Outlier
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

data_clean = remove_outlier(data_sel, 'FLIGHT_COUNT')
data_clean = remove_outlier(data_sel, 'avg_discount')

st.subheader('Remove Outlier')
st.markdown('<div style="text-align:justify;"> Data outlier is a data item that deviates significantly from the data and can cause analysis errors. So in this analysis, the outlier data in column ‘FLIGHT_COUNT’ and ‘AVG_DISCOUNT’ will be removed using the quartile calculation.</div>', unsafe_allow_html=True)
st.text("")

col1, col2, col5, col3, col4 = st.columns([2,3,1,3,1])
with col2:
    st.markdown('<b>Before Remove Outlier<b>', unsafe_allow_html=True)
with col3:
    st.markdown('<b>After Remove Outlier<b>', unsafe_allow_html=True)

col1, col2, col5, col3, col4 = st.columns([1,4,1,4,1])
with col2:
    feat_num = list(data_sel)
    fig = plt.figure(figsize=(15, 7))
    for i in range(0, len(feat_num)):
        plt.subplot(1, 7, i+1)
        sns.boxplot(y=data_sebelum[feat_num[i]],color='green',orient='v')
        plt.tight_layout()
    st.pyplot(fig)
with col3:
    fig = plt.figure(figsize=(15, 7))
    for i in range(0, len(feat_num)):
        plt.subplot(1, 7, i+1)
        sns.boxplot(y=data_clean[feat_num[i]],color='green',orient='v')
        plt.tight_layout()
    st.pyplot(fig)    


st.subheader('Correlation Matrix')
st.markdown('<div style="text-align:justify;"> We can see a linear relationship between the 2 variables. One of them, the correlation of the number of customer flights has a linear relationship with the total distance of flights made by customers, with a correlation of 0.89.</div>', unsafe_allow_html=True)
st.text("")
col1, col2, col5 = st.columns([1.5,3.5,2])
with col2:
    feat_num = list(data_sel)
    fig = plt.figure(figsize=(16, 10))
    corr_= data_clean.corr()
    sns.heatmap(corr_, annot=True, fmt = ".2f", cmap = "BuPu")
    st.pyplot(fig)


##SCALING
data_clean_up = data_clean.copy()
sc_data = StandardScaler()
data_std = sc_data.fit_transform(data_clean_up.astype(float))

st.header('Modelling')
st.subheader('Find the best K using Inertia')
st.markdown('<div style="text-align:justify;"> Find the best K for a dataset, using the Elbow method. Based on the Inertia Evaluation, find the point where the inertia drop starts to slow down. The elbow in this graph are in class 5, so in this model uses 5 class cluster.</div>', unsafe_allow_html=True)

inertia = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(data_std)
    inertia.append(kmeans.inertia_)

fig = plt.figure(figsize=(20, 10))
# plt.plot(inertia)
col1, col2, col5 = st.columns([1.5,3.5,2])
with col2:
    sns.lineplot(x=range(2, 11), y=inertia, color='#000087', linewidth = 4)
    sns.scatterplot(x=range(2, 11), y=inertia, s=300, color='#800000',  linestyle='--')
    st.pyplot(fig)


##CLUSTERING
kmeans = KMeans(n_clusters=5, random_state=0).fit(data_std)
df_data_std = pd.DataFrame(data=data_std, columns=list(data_clean_up))

df_data_std['clusters'] = kmeans.labels_
data_clean_up['clusters'] = kmeans.labels_
###########
#data_clean_up --> data cluster yang ditransformasi
#data_cluster --> data cluster yang ditransformasi balik
data_cluster = data_clean_up.copy()
data_cluster['MEMBERSHIP'] = np.expm1(data_cluster['MEMBERSHIP'])
data_cluster['LAST_TO_END'] = np.expm1(data_cluster['LAST_TO_END'])
data_cluster['FLIGHT_COUNT'] = np.expm1(data_cluster['FLIGHT_COUNT'])
data_cluster['SEG_KM_SUM'] = np.expm1(data_cluster['SEG_KM_SUM'])
data_cluster['avg_discount'] = np.expm1(data_cluster['avg_discount'])



st.header('Clustering & Visualisasi')

pca = PCA(n_components=2)

pca.fit(df_data_std)
pcs = pca.transform(df_data_std)

data_pca = pd.DataFrame(data = pcs, columns = ['PC 1', 'PC 2'])
data_pca['clusters'] = df_data_std['clusters']

data_pca['clusters'] = kmeans.labels_

st.subheader('Cluster Distribution')
st.markdown('<div style="text-align:justify;"> From the graph, we can see the distribution of the number of customers in each cluster.</div>', unsafe_allow_html=True)
col1, col2, col5 = st.columns([1.5,4,2])
with col2:
    f,ax = plt.subplots(1,1,figsize=(8,6))
    cluster = data_pca.groupby(['clusters'])['clusters'].count().reset_index(name='customers')
    color = sns.color_palette("RdPu", n_colors=10)
    color.reverse()
    grafik = sns.barplot(x='clusters', y ='customers', data=cluster, palette=['blue','red','green','orange','pink'])
    grafik.set_title('Cluster Distribution', weight='bold').set_fontsize('16')
    grafik.set_xlabel('Cluster').set_fontsize('13')
    grafik.set_ylabel('Customers').set_fontsize('13')
    st.pyplot(f)

st.subheader('Visualisasi Cluster')
st.markdown('<div style="text-align:justify;"> Clustering results are visualized using PCA so that clustering results are easier to visualize, and also reduce information that is not too important. So the segmentation of the clustering results can be seen.</div>', unsafe_allow_html=True)
col1, col2, col5 = st.columns([1.5,4,2])
with col2:
    fig, ax = plt.subplots(figsize=(15,10))
    sns.scatterplot(
        x="PC 1", y="PC 2",
        hue="clusters",
        edgecolor='green',
        linestyle='--',
        data=data_pca,
        palette=['blue','red','green','orange','pink'],
        s=160,
        ax=ax
    )
    st.pyplot(fig)


st.text("")

st.subheader('Cluster Radar Chart')
r1=pd.Series(kmeans.labels_).value_counts()
r2=pd.DataFrame(kmeans.cluster_centers_)
r3=pd.Series(['Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5'])

col1, col2, col5, col3, col4 = st.columns([0.5,4,0.5,4,0.5])
with col2:
    #Draw radar chart
    labels = np.array(['MEMBERSHIP','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','AVG_DISCOUNT'])#labels
    lab = ['','MEMBERSHIP','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','AVG_DISCOUNT']
    Length = 5
    r4=r2.T
    r4.columns=['MEMBERSHIP','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','AVG_DISCOUNT']
    fig = plt.figure()
    y=[]

    for x in ['MEMBERSHIP','LAST_TO_END','FLIGHT_COUNT','SEG_KM_SUM','AVG_DISCOUNT']:
        dt= r4[x]
        dt=np.concatenate((dt,[dt[0]]))
        y.append(dt)
    ax = fig.add_subplot(111, polar=True)

    angles = np.linspace(0, 2*np.pi, Length, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    ax.plot(angles, y[0], 'b-', linewidth=1.5)
    ax.plot(angles, y[1], 'r-', linewidth=1.5)
    ax.plot(angles, y[2], 'g-', linewidth=1.5)
    ax.plot(angles, y[3], 'y-', linewidth=1.5)
    ax.plot(angles, y[4], 'm-', linewidth=1.5)

    ax.legend(r3)
    ax.set_thetagrids(angles * 180/np.pi, lab)
    ax.set_title("Clustering Radar", va='bottom', fontproperties="sans serif")
    ax.set_theta_zero_location("N")
    ax.grid(True)
    st.pyplot(fig)
with col3:
    st.text("")
    st.text("")
    st.text("")
    st.dataframe(data_cluster.groupby('clusters').agg(['median']))

st.markdown('<div style="text-align:justify;"><b>Summary <br></b> 1.<b> Cluster 0 </b> is a customer who has been using flight services for a long time, for 6 years with the last flight booking distance of 105 days (approximately 3.5 months), with a total flight distance that has been carried out, namely 11,769 km with an average flight of 8 times . The average discount obtained is 68%. <br>1. <b> Cluster 1 </b>is a customer who has used flight services for 2.5 years, this cluster rarely makes flights because the median distance between the last flight and the last flight order is very long, which is 237 days or about once every 7 months with a total flight distance of 4,815 km and an average of 3 flights. Cluster 1 is a customer who rarely makes flights. The average discount obtained is 53%.<br>2. <b> Cluster 2 </b>is a customer who has used flight services for an average of 5 years and with a high intensity of making flights because the distance between the last flight and the last flight order was relatively fast, namely 13 days. The median total flight distance was 38,912 km with 28 flights. The average discount obtained is 73%<br>3. <b>Cluster 3 </b>is a customer who has used flight services for 2 years with a median distance of the last flight to the most recent flight order, which is 57 days. The total number of flights on average is 11 flights with a total distance of approximately 15,957 km. The average discount obtained is 70%.<br> 4. <b>Cluster 4 </b>is a customer who has used flight services for 2.8 years with a relatively short distance of 3,934 km. The median distance from the last flight to the last flight order, which is 296 days. This cluster is the cluster that rarely flies. Total number of flights on average 3 times. The average discount obtained is 81%.</div>', unsafe_allow_html=True)