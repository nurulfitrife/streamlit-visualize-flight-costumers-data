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
st.write("-----------------------------------------------------------------------------------------------")

st.subheader('I. Introduction')
st.markdown('<div style="text-align:justify;">In the current era of Big Data, numerous companies are implementing a data-driven approach by utilizing data to identify business trends, optimize marketing, analyze product sales, and make informed business decisions. All of these data-driven decisions and processes can be analyzed through data analysis. One such company that can benefit from data-driven approaches is an airline. By performing a characteristic analysis of customer data, an airline company can gain insights to develop a targeted marketing strategy and provide special treatment to specific customer segments. In this regard, unsupervised machine learning will be used to analyze data and cluster customers to determine customer segmentation.</div>', unsafe_allow_html=True)

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