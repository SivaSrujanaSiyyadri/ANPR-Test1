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


def header():
    st.markdown("<h3 style='text-align: center; color: white;'>Capstone Stone</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white;'>Automatic Number Plate Detection</h1>", unsafe_allow_html=True)
    st.markdown("![gif](https://cdn.discordapp.com/attachments/945603582462398464/948294399689912330/car-on-the-road-4851957-404227-unscreen.gif)")

def footer():
    st.subheader('About:')
    st.markdown('In this capstone project -ANPR , we have used OpenCv to detect the Licence plate in an image and give the converted string.\
    The pre processing of each frame or image is done using grayscale,canny-edge,invert and histogram equalization.\
    We have used the Pytesseract Library to perform OCR on the detected frame of licence plate.This then gives us the Licence Number Plate.')
    st.subheader('Techstack:')
    st.markdown('1)OpenCV: Helps with basic image processing not 100% accurate')
    st.markdown('2)PyTesseract: It is an Optical Character Recognising Problem (OCR). Therefore ,the Tesseract-OCR engine(pytesseract is the python implementation) helps in converting image to text.')
    st.markdown('3)Streamlit: Used to create web application for deployment and front end.')

def load_image(image_file):
    img = Image.open(image_file)
    return img
      
# Setting page layout
def config():
    st.set_page_config(
    page_title="Automatic Number Plate License Detection",  # Setting page title
    page_icon="🚗",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default   
    )

# Creating sidebar
def sidebar():
    with st.sidebar:
         st.header("Image Config")     # Adding header to sidebar
        # Adding file uploader to sidebar for selecting images
        source_img = st.file_uploader(
          "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    

# # Creating main page heading
# st.title("Automatic Number Plate License Detection")
# st.caption('Upload an image of a vehicle with a number plate.')
# st.caption('Then click the :blue[Detect License Plate] button and check the result.')

def yolomodel():
# Load Pre-trained ML Model
    model_path = Path(settings.DETECTION_MODEL)
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)


# Adding image to the first column if image is uploaded
def image():
    # Creating two columns on the main page
    col1, col2  = st.columns(2)
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
   
    try:
        text = result[0][-2]
    except Exception as e:
        text = "No Text Detected"
      
    try:
        st.write("Detected License Plate:", text)
    except Exception as e:
        st.write("No License Plate Detected")


def main():
    config()
    header()
    add_selectbox = st.sidebar.selectbox(
    "What do you want to upload?",
    ("Image", "Video"))
    if add_selectbox=='Image':
        sidebar()
        yolomodel()
        image()
    if add_selectbox=='Video':
        yolomodel()
        video()
    footer()
        
if __name__ == '__main__':
    main()