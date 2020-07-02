import cv2 
import streamlit as st 
from PIL import Image,ImageEnhance
import numpy as np 
import os


face_cascade = cv2.CascadeClassifier('harc/face.xml')

def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img= cv2.cvtColor(new_img,1)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #detect
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    #draw rect
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img,faces


def main():
    st.title("New APP")
    st.text("check this out")

    activities = ['detection', 'about']
    choise = st.sidebar.selectbox('select', activities)

    if choise == 'detection':
        st.subheader("face detection")
        image_file =st.file_uploader("upload", type =['jpg','png','jpeg'])
        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("org image")
            st.image(our_image)
            
        enhance_type = st.sidebar.radio("Enhance type",["Original","gray","contrast","brightness","blur"])
        
        if enhance_type == "gray":
            new_img = np.array(our_image.convert('RGB'))
            img= cv2.cvtColor(new_img,1)
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            st.write("converted image")
            st.image(gray)

        if enhance_type == "contrast":
            c_rate = st.sidebar.slider("contrast",0.5,3.5)
            enhancer = ImageEnhance.Contrast(our_image)
            img_out = enhancer.enhance(c_rate)
            st.write("new image")
            st.image(img_out)
        
        if enhance_type == "brightness":
            c_rate =st.sidebar.slider("brightness",0.5,3.5)
            enhancer =ImageEnhance.Brightness(our_image)
            img_out =enhancer.enhance(c_rate)
            st.write("new image")
            st.image(img_out)


        if enhance_type == "blur":
            new_img = np.array(our_image.convert('RGB'))
            blur_rate = st.sidebar.slider("blur",0.5,3.5)
            img= cv2.cvtColor(new_img,1)
            blur= cv2.GaussianBlur(img,(11,11),blur_rate)
            st.write("blured image")
            st.image(blur)


        task = ["faces","smiles", "eyes", "cannise", "cartonize"]
        feature_choise = st.sidebar.selectbox("find features", task)

        if st.button("process"):
                if feature_choise == "faces":
                        result_img,result_faces = detect_faces(our_image)
                        st.image(result_img)
                        st.success("found {} faces".format(len(result_faces)))      	

    elif choise == 'about':
        st.subheader("about")


if __name__=='__main__':
    main()
