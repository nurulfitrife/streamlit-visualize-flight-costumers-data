import streamlit as st
import pickle
from PIL import Image
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Flight Customers Analysis", layout="wide"
)
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

st.title("Flight Customers Analysis")
st.write("----------------------------------------------------------------------------------------------")

st.subheader('I. Introduction')
st.markdown('<div style="text-align:justify;">Pada era Big Data saat ini, banyak perusahaan yang menerapkan data driven yang mana perusahaan melibatkan data untuk melihat tren bisnis, memaksimalkan pemasaran, menganalisis penjualan produk perusahaan, hingga menjadi faktor untuk mengambil keputusan bisnis. Semua keputusan dan proses data driven tersebut dapat dianalisis dari data. Salah satu perusahaan yang dapat menerapkan data driven, yaitu perusahaan penerbangan. Dengan mengetahui analisis karakteristik dari customer data, suatu perusahaan penerbangan dapat mendapatkan insight untuk merencanakan marketing strategy, dan juga dapat memaksimalkan pemasaran ataupun memberikan perlakuan khusus bagi karakteristik tertentu. Pada kesempatan ini akan dilakukan analisis data serta clustering untuk mengetahui segmentasi customers menggunakan Unsupervised Machine Learning.</div>', unsafe_allow_html=True)

st.write("")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    img = Image.open('data/flightpath.png')
    st.image(img, caption='source:https://x96.com/wp-content/uploads/2020/02/flightpath.png')

st.write("")
st.subheader('II. Metadata Data')
col1, col2, col3 = st.columns([1,2,1])
with col2:
    img = Image.open('data/metadata.png')
    st.image(img, caption='Metadata Flight Data')


st.write("")
st.subheader('III. Data Reference')
st.markdown('flight-data: https://www.programmersought.com/article/48113472881/')