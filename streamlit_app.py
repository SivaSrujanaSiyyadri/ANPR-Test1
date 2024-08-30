# Python In-built packages
from pathlib import Path
import PIL
import tempfile
import numpy as np
import easyocr

# External packages
import streamlit as st
import cv2
# import numpy as np

# Local Modules
import settings
import helper


# Setting page layout
st.set_page_config(
    page_title="Automatic Number Plate License Detection",  # Setting page title
    page_icon="🚗",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default   
)

# Creating sidebar
with st.sidebar:
    st.header("Image Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    

# Creating main page heading
st.title("Automatic Number Plate License Detection")
st.caption('Upload an image of a vehicle with a number plate.')
st.caption('Then click the :blue[Detect License Plate] button and check the result.')
# Creating two columns on the main page
col1, col2 , col3= st.columns(3)

# Load Pre-trained ML Model
model_path = Path(settings.DETECTION_MODEL)
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)


# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        print(uploaded_image)
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                 use_column_width=True
                 )
        

if st.sidebar.button('Detect License Plate'):
    if source_img is None:
        st.warning("Please upload an image.")
        st.stop()

    # Load the image
    uploaded_image = PIL.Image.open(source_img)

    res = model.predict(uploaded_image)
    #st.text(res)
    boxes = res[0].boxes
    print('boxes', boxes)
    res_plotted = res[0].plot()[:, :, ::-1]
    st.image(res_plotted, caption='Detected Image',
                use_column_width=True)
        # # Read image
    img_array = np.array(uploaded_image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    print('Boxes xyxy:   ',boxes.xyxy.tolist()[0])
    x1, y1, x2, y2 = boxes.xyxy.tolist()[0]
    # Crop the object using the bounding box coordinates
    cropped_image = gray[int(y1):int(y2), int(x1):int(x2)]
    st.image(cropped_image, caption='Croped Image',
                use_column_width=True)
    print('---------------------------------------------')
    
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    print('Result :',result)
   
    with col3:
        try:
            text = result[0][-2]
        except Exception as e:
            text = "No Text Detected"
      
        try:
            st.write("Detected License Plate:", text)
        except Exception as e:
            st.write("No License Plate Detected")

