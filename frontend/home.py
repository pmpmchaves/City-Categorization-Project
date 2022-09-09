from hashlib import new
from PIL import Image
import streamlit as st
from city_categorization.main import image_load
from scripts.get_image import get_city_info
from city_categorization import main
import numpy as np
import time

##############################################
## Setting page configurations on Streamlit ##
##############################################
st.set_page_config(
            page_title="City Categorization",
            page_icon="🌎",
            layout="centered", # wide
            initial_sidebar_state="auto") # collapsed

LEGEND_WIDTH = 150
CSS = """
h1, h2 {
    text-align: center;
}
h4 {
    text-align: center;
}
h6 {
    text-align: center;
    font-size: 16px;
    font-weight: 100;
}
p {
    text-align: center;
}
"""

st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

st.markdown("""# City Categorization
## A pilot project to assist urban planning actions 🗺
###### made with 😀 by **Francisco Garcia**, **Pedro Chaves** and **Rodrigo Pinto** in 🇵🇹""")
st.markdown("""---""")
###############
## Main code ##
###############

#Columns
columns = st.columns(2)

uploaded_file = columns[0].file_uploader("Choose a file")
city = columns[1].text_input('City name 🌎')

if columns[1].button('Click me to show the city satellite image'):
    if city:
        st.markdown("""---""")
        city_info, coords = get_city_info(city=city)
        st.write(f"Searching for **{city_info[0]['name']}** 🗺. Its population is about **{city_info[0]['population']:,.2f}**")
        st.write('Working to find the best image...')
        st.write('🔎 🌍')
        city_image = image_load(city=city)
        time.sleep(2)
        st.success('The search is over! 🎯 🏆')
        st.markdown("""---""")
        if city_image:
            new_columns = st.columns(2)
            image = Image.open(f'{city}.tiff')
            new_columns[0].image(image, caption=city.capitalize())
            #with st.spinner('Loading image, running the machine learning model 🤖 and generating the output 🖨...'):
            final_df, final_image = main.final_outputs(city)
            processed_image = Image.fromarray(final_image.astype(np.uint8))
            new_columns[1].image(final_image, caption='Processed Image')
            results_columns = st.columns(2)
            results_columns[0].table(final_df)
            results_columns[1].image('frontend/legend.jpg', width=LEGEND_WIDTH)
    elif uploaded_file is not None:
        st.markdown("""---""")
        new_columns = st.columns(2)
        new_columns[0].image(uploaded_file, caption=str(uploaded_file.name).split('.')[0].capitalize())
        #with st.spinner('Loading image, running the machine learning model 🤖 and generating the output 🖨...'):
        final_df, final_image = main.final_outputs(str(uploaded_file.name).split('.')[0].capitalize())
        processed_image = Image.fromarray(final_image.astype(np.uint8))
        new_columns[1].image(final_image, caption='Processed Image')
        results_columns = st.columns(2)
        results_columns[0].table(final_df)
        results_columns[1].image('frontend/legend.jpg', width=LEGEND_WIDTH)



#     with st.spinner('Working in deep learning'):
#         final_df, final_image = main.final_outputs(city)
#         processed_image = Image.fromarray(final_image.astype(np.uint8))
#         processed_image.save(f'{city}_categorized.tiff')

#         #Columns for results
#         results_columns = st.columns(2)
#         results_columns[0].table(final_df)
#         results_columns[1].image(final_image, caption='Processed Image')
