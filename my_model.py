import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit as st
from utils_sign_rec import preprocess_landmark, get_landmarks_list, return_alphabet
import string
import pickle
import urllib.request

# map numbers to alphabets
alphabet = list(string.ascii_uppercase)
dict_alphabet = {}
for i in range(26):
    dict_alphabet[i] = alphabet[i]

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# For static images:
mp_model = mp_hands.Hands(
    static_image_mode=True,  # only static images
    max_num_hands=1,  # max 1 hand detection
    min_detection_confidence=0.5,  # detection confidence
    min_tracking_confidence=0.7,  # confidence for interpolating landmarks during movement
)


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_model_from_github():
    # Download the pickle model file from GitHub
    url = "https://raw.githubusercontent.com/bazzimahdi/asl-sign-language-recognition-mediapipe/master/svm_model.pkl"
    model_path = "svm_model.pkl"  # Path to save the downloaded model file
    urllib.request.urlretrieve(url, model_path)

    # Load the model from disk
    clf = pickle.load(open(model_path, "rb"))
    return clf


@st.cache(suppress_st_warning=True)
def run_object_detection():
    clf = load_model_from_github()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("No webcam detected. Please make sure a webcam is connected.")
        return

    stframe = st.empty()  # Placeholder for displaying the video

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

            threshold = 0.3
            if logit > threshold:
                cv2.putText(
                    roi,
                    dict_alphabet[pred[0]],
                    (50, 75),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    (0, 250, 200),
                    2,
                )
                stframe.image(roi, channels="BGR")
            else:
                cv2.putText(
                    roi,
                    "no sign detected",
                    (50, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 250, 200),
                    2,
                )
                stframe.image(roi, channels="BGR")

        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    st.title("Real-Time Object Detection")
    st.write("This is a real-time object detection application using Streamlit.")
    st.write(
        "It uses mediapipe for landmarks extraction and a pre-trained svm model to detect hand signs."
    )

    if st.button("Run Object Detection"):
        run_object_detection()


if __name__ == "__main__":
    main()
