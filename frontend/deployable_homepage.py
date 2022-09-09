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
    page_title="City Typologies",
    page_icon="🌎",
    layout="centered",  # wide
    initial_sidebar_state="auto")  # collapsed

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

st.markdown("""# Computer Vision: City Typology Classification
## Pilot Project to Support Urban Planning 🗺
###### Made with ❤️ by **Francisco Garcia**, **Pedro Chaves** and **Rodrigo Pinto** in 🇵🇹"""
            )
st.markdown("""---""")
###############
## Main code ##
###############

#Columns
columns = st.columns(2)

uploaded_file = columns[0].file_uploader("Choose a File")
if uploaded_file is not None:
    columns[1].image(uploaded_file,
                     caption=str(uploaded_file.name).capitalize())

columns[0].write('#### OR')

city = columns[0].text_input('City Name 🌎')
if columns[0].button('Click Me to Find Satellite Image'):
    city_info, coords = get_city_info(city=city)
    columns[0].write(
        f"Searching for **{city_info[0]['name']}** 🗺. The Population of {city} is **{city_info[0]['population']:,.2f}**"
    )
    columns[0].write('Working to Find the Best Image...')
    columns[0].write('🔎 🌍')
    city_image = image_load(city=city)
    columns[0].success('The Search is Over! 🎯 🏆')
    time.sleep(2)
    image = Image.open(f'{city}.tiff')
    columns[1].image(image, caption=city.capitalize())
    st.markdown("""---""")

    with st.spinner('Running Deep Learning Model'):
        final_df, final_image = main.final_outputs(city)
        processed_image = Image.fromarray(final_image.astype(np.uint8))
        processed_image.save(f'{city}_categorized.tiff')

        #Columns for results
        results_columns = st.columns(2)
        results_columns[0].dataframe(final_df)
        results_columns[1].image(final_image, caption='Processed Image')
