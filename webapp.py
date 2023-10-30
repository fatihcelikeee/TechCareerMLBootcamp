import base64
import streamlit as st
from PIL import Image
import numpy as np
import streamlit as st
import pickle
from skimage.transform import resize
import tempfile

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)  

def classify(image, model, categories):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image.save(temp_file.name)

# Read the temporary file with skimage
    numpy_image = np.array(image)
    data = []
    img = resize(numpy_image, (15, 15))
    data.append(img.flatten())
    data = np.asarray(data)
    y_prediction = model.predict(data)
    y_prediction = categories[y_prediction[0]]
    prediction_text = f'Category: {y_prediction}'
    return prediction_text

set_background("./bg.jpg")
# set title
st.title('Wildfire classification')

# set header
st.header('Please upload a forest image')

# upload file
file = st.file_uploader('Upload a forest image', type=['jpeg', 'jpg', 'png'])  # Provide a non-empty label here

with open("trained_model.pkl", 'rb') as model_file:
        model = pickle.load(model_file)

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    #classify image
    class_name = classify(image, model, ["fire","nofire"])

    # write classification
    st.write("## {}".format(class_name))