import cv2
import imutils
from PIL import Image
import streamlit as st
import numpy as np
import tempfile
import pytesseract
import easyocr
# import tesseract-ocr
from pathlib import Path
# import re

import settings
import helper



pytesseract.pytesseract.tesseract_cmd ='tesseract'
model=None
global invert

def header():
    st.markdown("<h1 style='text-align: center; color: white;'>Automatic Number plate detection</h1>", unsafe_allow_html=True)
    
def load_image(image_file):
    img = Image.open(image_file)
    return img

def footer():
    st.subheader('About:')
    st.markdown('In this Deep Learning Application, we have used OpenCv to detect the Licence plate in an image and give the converted string.\
    The pre processing of each frame or image is done using grayscale,canny-edge,invert and histogram equalization.\
    We have used the Pytesseract Library to perform OCR on the detected frame of licence plate.This then gives us the Licence Number Plate.')
    st.subheader('Techstack:')
    st.markdown('1)OpenCV: Helps with basic image processing not 100% accurate')
    st.markdown('2)PyTesseract: It is an Optical Character Recognising Problem (OCR). Therefore ,the Tesseract-OCR engine(pytesseract is the python implementation) helps in converting image to text.')
    st.markdown('3)Streamlit: Used to create web application for deployment and front end.')
    st.subheader('Conclusion:')
    st.markdown('There is an immediate need of such kind of Automatic Number Plate Recognition system in India as there are problems of traffic, stolen cars etc.\
    The Government should take some interest in developing this system as it is very economical and eco-friendly, if applied effectively.\
    This change will help in the progress of the nation.')

# Load Pre-trained ML Model
def trainmodel():
    global model
    model_path = Path(settings.DETECTION_MODEL)
    try:
        model = helper.load_model(model_path)
        return model
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

# def isValidVehicleNumberPlate(str):
 
#     # Regex to check valid Indian Vehicle Number Plate
#     regex = "^[A-Z]{2}[\\s-]{0,1}[0-9]{2}[\\s-]{0,1}[A-Z]{1,2}[\\s-]{0,1}[0-9]{4}$"
     
#     # Compile the ReGex
#     p = re.compile(regex)
#     print(p)
 
#     # If the string is empty
#     # return false
#     if (str == None):
#         return False
 
#     # Return if the string
#     # matched the ReGex
#     if(re.search(p, str)):
#         return True
#     else:
#         return False

def video():
    #st.subheader('Choose a photo')
    Video = st.file_uploader('',type=['mp4'])
    #st.image(load_image(images))
    #Function to read the image  
    #image = cv2.imread(image) 
    # Function to resize the image  
    if Video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(Video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        i=0
        while(cap.isOpened()):
            ret,frame = cap.read()
            i=i+1
            if ret==True and i%285==0:
                st.spinner("Please wait")
                image = imutils.resize(frame,height=620,width=480)
            #Convert BGR image to GRAYSCALE
                image1=image.copy()
                gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                #Canny edge detection to detect edges in an image 
                edged=cv2.Canny(gray_image,30,200)
                # Function to join similar edges 
                #Take two arguments : 
                #First:: It take all the contours but doesn't create parent-child relationship
                #Second:: It specify to take corner points off the counter
                cnts,new = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)   
                #This function is used to create counters based to given counter cordinates
                
                cv2.drawContours(image1,cnts,-1,(0,255,0),3)
                #Here we are sorting the conters and neglecting all the conters whose area is less than 20
                cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:20]
                screenCnt = None
                for c in cnts:
                    #To determine sqaure cure among the identified contours
                    perimeter = cv2.arcLength(c,True)
                    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
                    if len(approx) == 4: 
                        screenCnt = approx
                        x,y,w,h = cv2.boundingRect(c) 
                        frame=cv2.rectangle(image,(x,y),(x+w,y+h),(255, 0, 0),2)
                        new_img=image[y:y+h,x+3:x+w+3]
                        resized = cv2.resize(new_img,dsize=None,fx=4,fy=4)
                        invert = cv2.bitwise_not(resized)
                        gray_image1= cv2.cvtColor(invert,cv2.COLOR_BGR2GRAY)
                        gray_image1=cv2.equalizeHist(gray_image1)
                        threshold=cv2.threshold(gray_image1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                        #rem_noise=cv2.medianBlur(threshold,5)
                        #PyTesseract to convert text in the image to string
                        plate = pytesseract.image_to_string(threshold, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8')


                        # reader = easyocr.Reader(['en'],model_storage_directory='.')
                        # try:
                        #     plate = reader.readtext(threshold,paragraph="False")[0][1]
                        # except Exception as e:
                        #     plate = "No image Detected"
                        
                        if(plate==''):
                            plate="Error"
                        if(len(plate)>10): #Car number plate can not have more than 10 characters 
                            st.subheader("Number plate is : "+plate[:13])
                            st.balloons() 
                            break 
                        else:
                            st.subheader("Number plate is : "+plate) 
                            st.balloons() 
                            break
                        break
                
                
                cv2.drawContours(image,[screenCnt],-1,(0,255,0),2)
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break 
            stframe.image(frame)
                            

def image():
    images = st.file_uploader('',type=['jpeg','png', 'jpg'])   
    if images is not None:
        file_bytes = np.asarray(bytearray(images.read()), dtype=np.uint8)
        image=cv2.imdecode(file_bytes,1)
        image = imutils.resize(image,height=620,width=480)
        # uploaded_image = PIL.Image.open(image)
        
        res = model.predict(image)
        #st.text(res)
        boxes = res[0].boxes
        print('boxes', boxes)
        res_plotted = res[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption='Detected Image',
                use_column_width=True)
        # Read image
        

        #Convert BGR image to GRAYSCALE
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #Canny edge detection to detect edges in an image 
        edged=cv2.Canny(gray_image,30,200)

        print('Boxes xyxy:   ',boxes.xyxy.tolist()[0])
        x1, y1, x2, y2 = boxes.xyxy.tolist()[0]
        # Crop the object using the bounding box coordinates
        cropped_image = gray_image[int(y1):int(y2), int(x1):int(x2)]
        # st.image(cropped_image, caption='Croped Image',
        #         use_column_width=True)
        # Function to join similar edges 
        #Take two arguments : 
        #First:: It take all the contours but doesn't create parent-child relationship
        #Second:: It specify to take corner points off the counter
        cnts,new = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)   
        image1=cropped_image.copy()
        #This function is used to create counters based to given counter cordinates
        cv2.drawContours(image1,cnts,-1,(0,255,0),3)
        #Here we are sorting the conters and neglecting all the conters whose area is less than 20
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:20]
        screenCnt = None
        image2 = image.copy()
        #---------------
        
        image_test = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        gray_image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
        #------------------
        for c in cnts:
            #To determine sqaure cure among the identified contours
            perimeter = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
            print('Length of Approx: ',len(approx))
            if len(approx) == 4: 
                screenCnt = approx
                # x,y,w,h = cv2.boundingRect(c)
                new_img=image[int(y1):int(y2), int(x1):int(x2)]
                # new_img=image[y:y+h,x:x+w]
                # st.image(new_img, caption='New Image',
                # use_column_width=True)
                resized = cv2.resize(new_img,dsize=None,fx=4,fy=4)
                invert = cv2.bitwise_not(resized)
                gray_image1= cv2.cvtColor(invert,cv2.COLOR_BGR2GRAY)
                gray_image1=cv2.equalizeHist(gray_image1)
                threshold=cv2.threshold(gray_image1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                #rem_noise=cv2.medianBlur(threshold,5)
                #PyTesseract to convert text in the image to string
                # plate = pytesseract.image_to_string(threshold, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8')
                # if(plate==''):
                #     plate="Error"
                # if(len(plate)>10): #Car number plate can not have more than 10 characters 
                #     st.subheader("Number plate detected is : "+plate[:13])  
                #     #st.balloons()
                # else:
                #     st.subheader("Number plate detected is : "+plate)  
                #     #st.balloons
                # break 
                reader = easyocr.Reader(['en'])
                result = reader.readtext(cropped_image)
                print('Result :',result)      
        cv2.drawContours(image,[screenCnt],-1,(0,255,0),2) 
        col1, col2= st.columns(2)
        with col1:
            st.image(gray_image,caption='Gray-scale image',width=300)
        with col1:
            st.image(cropped_image, caption='Croped Image',
             use_column_width=True)
        with col2:
            st.image(edged,caption='Canny Edge',width=300)
        col1, col2, col3= st.columns(3)
        with col1:
            st.image(invert,caption='Inverted B & W',width=150)
        with col2:
            st.image(gray_image1,caption='Grayscale Image',width=150)
        with col3:
            st.image(threshold,caption='Threshold Image',width=150)  
        with col1:
            try:
                print('Result value: ',result)
                text = result[0][-2]
                # text = result[0][1]
                # License_number=isValidVehicleNumberPlate(text)
            except Exception as e:
                text = "No Text Detected"
      
            try:
                st.write("Detected License Plate:", text)
            except Exception as e:
                st.write("No License Plate Detected")  
            
        
def main():
    header()
    model=trainmodel()
    add_selectbox = st.sidebar.selectbox(
    "What do you want to upload?",
    ("Image", "Video"))
    if add_selectbox=='Image':
        image()
    if add_selectbox=='Video':
        video()
    footer()
        
if __name__ == '__main__':
    main()