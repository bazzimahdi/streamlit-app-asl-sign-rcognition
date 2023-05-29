import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit as st
from utils_sign_rec import preprocess_landmark, get_landmarks_list, return_alphabet
import pickle
import urllib.request


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# For static images:
mp_model = mp_hands.Hands(
    static_image_mode=True,  # only static images
    max_num_hands=1,  # max 1 hand detection
    min_detection_confidence=0.5,  # detection confidence
    min_tracking_confidence=0.7,  # confidence for interpolating landmarks during movement
)


def load_model_from_github():
    # Download the pickle model file from GitHub
    url = "https://raw.githubusercontent.com/bazzimahdi/asl-sign-language-recognition-mediapipe/master/svm_model.pkl"
    model_path = "svm_model.pkl"  # Path to save the downloaded model file
    urllib.request.urlretrieve(url, model_path)

    # Load the model from disk
    clf = pickle.load(open(model_path, "rb"))
    return clf


def run_from_video(video_path, thresh, clf):
    
    cap = cv2.VideoCapture(video_path)
    i = 0  # frame counter
    stframe = st.empty()
    while cap.isOpened():
        ret = cap.grab()  # grab frame
        if ret:
            i = i + 1  # increment counter
            if i % 3 == 0:  # display only one third of the frames, you can change this parameter according to your needs // video is 20 fps and processing is ~6 fps
                ret, frame = cap.retrieve()  # decode frame
                roi_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mp_model.process(roi_RGB)
                stframe.image(frame)
                if results.multi_hand_landmarks:
                    landmarks_data = []
                    # preprocess landmarks to be compatible with svm clf
                    landmarks = preprocess_landmark(
                        get_landmarks_list(roi_RGB, results), False
                    )
                    landmarks_data.append(landmarks)
                    # extract and plot the landmarks
                    for handLms in results.multi_hand_landmarks:
                        cx_data = []
                        cy_data = []
                        for id, lm in enumerate(handLms.landmark):
                            h, w, c = roi_RGB.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cx_data.append(cx)
                            cy_data.append(cy)
                            cv2.circle(roi_RGB, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
                        mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                        pred = clf.predict(landmarks_data)
                        logit = max(clf.predict_proba(landmarks_data)[0])
                        if logit > thresh:
                            cv2.putText(frame,return_alphabet(pred[0]),(50, 75),cv2.FONT_HERSHEY_PLAIN,3,(0, 250, 200),2)  # return alphabet from previous section
                        else :
                            cv2.putText(frame,"no sign detected",(50, 50),cv2.FONT_HERSHEY_PLAIN,2,(0, 250, 200),2)
                stframe.image(frame, channels="BGR")

        else:
            break



def run_object_detection(thresh, clf):
    
    stframe = st.empty()  # Placeholder for displaying the video
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("No webcam detected. Please make sure a webcam is connected.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            st.warning("Failed to retrieve frame from the webcam.")
            break

        ##############################
        frame = cv2.flip(frame, 1)

        roi = frame.copy()
        stframe.image(roi, channels="BGR")

        results = mp_model.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks_data = []
            # preprocess landmarks to be compatible with svm clf
            landmarks = preprocess_landmark(get_landmarks_list(roi, results), False)
            landmarks_data.append(landmarks)

            for handLms in results.multi_hand_landmarks:
                cx_data = []
                cy_data = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = roi.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(roi, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
                mp_drawing.draw_landmarks(roi, handLms, mp_hands.HAND_CONNECTIONS)

            pred = clf.predict(landmarks_data)
            logit = max(clf.predict_proba(landmarks_data)[0])

            if logit > thresh:
                cv2.putText(roi,return_alphabet(pred[0]),(50, 75),cv2.FONT_HERSHEY_PLAIN,3,(0, 250, 200),2)
                stframe.image(roi, channels="BGR")
            else:
                cv2.putText(roi,"no sign detected",(50, 50),cv2.FONT_HERSHEY_PLAIN,2,(0, 250, 200),2)
                stframe.image(roi, channels="BGR")

        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()



custom_styles = """
<style>
/* Change background color of h1 element */
h1 {
    background-color: #8FBC8F;
    text-align: center
    }

/* Change background color of h2 element */
h2 {
        text-align: center
    }
    
/* Customize button style */
.stButton {
    background-color: #F0F0F0;
    color: black;
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
} 
</style>
"""



def main():
    clf = load_model_from_github()
    st.set_page_config(page_title='Sign Language Recognition',layout='wide',page_icon= 'icon.png')

    st.markdown(custom_styles,unsafe_allow_html=True)

    # st.markdown(h6_style, unsafe_allow_html=True)


    st.write("# American Sign Language Finger Spelling")
    st.write("")
    st.image("asl.jpeg",use_column_width=True)
    st.write("## The model is based on mediapipe Hands pose estimation model for landmark extraction and SVM for classification")
    st.write("")
    st.write("")
    st.write("")
    st.write("--------------------------------------------------------------------------------------------------------------------------------")

    st.write(" Set a minimum detection threshold")
    threshold = st.slider("", 0.3, 1.0, 0.5)
    st.write("")
    st.write("--------------------------------------------------------------------------------------------------------------------------------")


    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
    if uploaded_file:
        
        video_bytes = uploaded_file.read()
        temp_file_path = "temp_video.mp4"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(video_bytes)

        st.write("upload done!")
        if st.button('process video'):
            run_from_video(temp_file_path,thresh=threshold, clf)
            
    st.write("")
    st.write("")
    st.write("--------------------------------------------------------------------------------------------------------------------------------")
    st.write("Run the model in Real-time")
    if st.button("Run Object Detection"):
        run_object_detection(thresh=threshold, clf)

if __name__ == "__main__":
    main()
