# Python In-built packages
from pathlib import Path
import PIL
import tempfile

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
col1, col2 = st.columns(2)

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
    res_plotted = res[0].plot()[:, :, ::-1]
    st.image(res_plotted, caption='Detected Image',
                use_column_width=True)
        # # Read image
    # Save the uploaded image to a temporary file and read it
    tfile = tempfile.NamedTemporaryFile(delete=True)
    print('tfile ',tfile.name)
    print(source_img.read())
    tfile.write(source_img.read())
    img = cv2.imread(tfile.name)

    print('img' , img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cropped_image = gray[topx:bottomx+1, topy:bottomy+1]


    # # Use Easy OCR to read text
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    tfile = tempfile.NamedTemporaryFile(delete=True)
    tfile.write(source_img.read())
    img = cv2.imread(tfile.name)
    with col2:
        try:
            text = result[0][-2]
        except Exception as e:
            text = "No Text Detected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
        res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
        st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), caption="Detected License Plate", use_column_width=True)

        try:
            st.write("Detected License Plate:", text)
        except Exception as e:
            st.write("No License Plate Detected")


    # render yolo image as well
    # r_img = res.render() # returns a list with the images as np.array
    # img_with_boxes = r_img[0] # image with boxes as np.array
    # st.image(img_with_boxes, caption='Yolo Image',
    #             use_column_width=True)
    # # Save the uploaded image to a temporary file and read it
    # tfile = tempfile.NamedTemporaryFile(delete=True)
    # tfile.write(source_img.read())

    # # Read image
    # img = cv2.imread(tfile.name)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # Apply filter and find edges for localization
    # bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    # edged = cv2.Canny(bfilter, 30, 200) #Edge detection

    # # Find contours and apply mask
    # keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = imutils.grab_contours(keypoints)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # location = None
    # for contour in contours:
    #     approx = cv2.approxPolyDP(contour, 10, True)
    #     if len(approx) == 4:
    #         location = approx
    #         break

    # mask = np.zeros(gray.shape, np.uint8)
    # new_image = cv2.drawContours(mask, [location], 0,255, -1)
    # new_image = cv2.bitwise_and(img, img, mask=mask)


    # # Crop license plate
    # (x,y) = np.where(mask==255)
    # (topx, topy) = (np.min(x), np.min(y))
    # (bottomx, bottomy) = (np.max(x), np.max(y))
    # cropped_image = gray[topx:bottomx+1, topy:bottomy+1]


    # # Use Easy OCR to read text
    # reader = easyocr.Reader(['en'])
    # result = reader.readtext(cropped_image)

    # with col2:
    #     try:
    #         text = result[0][-2]
    #     except Exception as e:
    #         text = "No Text Detected"
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    #     res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
    #     st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), caption="Detected License Plate", use_column_width=True)

    #     try:
    #         st.write("Detected License Plate:", text)
    #     except Exception as e:
    #         st.write("No License Plate Detected")

'''
# Setting page layout
st.set_page_config(
    page_title="Number Plate Detection using YOLOv5",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Number Plate Detection using YOLOv5")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
'''