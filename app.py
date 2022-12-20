from PIL import Image
from os.path import join, dirname, realpath
from glob import glob
import tensorflow as tf
import tensorflow.keras.utils as image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

import streamlit as st
import numpy as np
import tensorflow as tf
import numpy as np

import os
import cv2
import imutils
import math


# Loading images to frontend
def load_image(image_file):
	img = Image.open(image_file)
	return img


model = tf.keras.applications.InceptionV3(weights='imagenet')

st.write("Welcome to Video Checker")
st.write('Plase follow the following instructions')
st.write("1. Kindly change the name of the video to sample1 before you upload ")
st.write("2. Put in the word you would love to search in the search query box")
st.write("3. Wait for the video to process and send you another video of the searched word")

# User search query


# Allow user to upload video
video = st.file_uploader(label="upload video", type="mp4", key="video_upload_file")

# Continue only if video is uploaded successfully
if(video is not None):
    # Notify user
    st.text("video has been uploaded")
    # Gather video meta data
    file_details = {"filename":video.name, "filetype":video.type,
                    "filesize":video.size}
    # Show on ui
    st.write(file_details)
    # save video
    with open(video.name, "wb") as f:
        f.write(video.getbuffer())
    # Notify user
    st.success("file saved")

    # Show video on ui 
    video_file = open(file_details['filename'], 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
	
    search_query = st.text_input("enter object to query","" )

	
    

    # Create frames for the video and save 
    def create_frames():
        images_array = []
        cap = cv2.VideoCapture(video.name)
        index = 0

        while True:
            ret, frame = cap.read()
            if ret == False:
                cap.release()
                break
            if frame is None:
                break
            else:
                if index == 0:
                    images_array.append(frame)
                    cv2.imwrite(f"frames/{index}.jpeg", frame)

                else:
                    if index % 10 == 0:
                        images_array.append(frame)
                    cv2.imwrite(f"frames/{index}.jpeg", frame)

            index += 1
        return np.array(images_array)
    

    # Create frames
    images_array = create_frames()

    # Continue only if frames have been successfully created 
    if len(images_array) > 0:
        frame_paths = glob(f"frames/*.jpeg")

    cap = cv2.VideoCapture('sample1.mp4') #capturing the video from given path
    List =[]
    top_result = None
    frameRate = cap.get(5) #frame rate
     
    x=1
    count = 0
    inPath = ""
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename = "frames/*.jpeg"
            cv2.imwrite(filename, frame)
            # img_path = ''
            img = image.load_img(filename, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # predict class
            preds = model.predict(x)

            # get the top 10 classes
            top3 = decode_predictions(preds, top=100)[0]

        # print top 10 classes
            for result in top3:
                if search_query == result[1]:
                    inPath = filename
                    img_array = []
                    img = cv2.imread(filename)
                    height, width, layers = img.shape
                    size = (width,height)
                    img_array.append(img)
                    out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
                    for i in range(len(img_array)):
                        out.write(img_array[i])
                    out.release()
    video_file = open('project.mp4', 'rb') #enter the filename with filepath
    video_bytes = video_file.read() #reading the file
    st.video(video_bytes) #displaying the video

         
    # st.image(load_image('project.mp4'),width=250)


    
    # Resize the frames for the model
    def resize_frames():
        frame_paths = glob(f"frames/*.jpeg")
        index = 0
        width, height = (299, 299)

        for frame in frame_paths:
            image = cv2.imread(frame)
            image_resized = cv2.resize(image, (299, 299))
            cv2.imwrite("resized/%i.jpeg"%index, image_resized)
            
            index += 1  

    # Resize the frames for the model 
    resize_frames()

    # Classify frames
    def predict():
        def fetch_frames():
            frame_paths = glob(f"resized/*.jpeg")
            query_frames_array = []

            for frame in frame_paths:
                image = cv2.imread(frame)
                image = np.expand_dims(image, axis=0)
                query_frames_array.append(image)
                
            return np.array(query_frames_array)

    
