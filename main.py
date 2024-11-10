import cv2
from cv2.typing import MatLike
import face_recognition
import os
import pickle
import uuid
import tkinter.messagebox
import time
from threading import Thread

from pathlib import Path

from modules.face_data import FaceData

SAVE_FLAG = False    # Do we need to save data (train)?

SCALE_KOEFF = 0.5   # Scale of video input

FACES_ENCODS = 10    # How much encods save and use
SIMILAR_FACES_DIST = 0.3    # How close must be characteristic vectors to each other to count them as same
RECOG_TOLERANCE = 0.5   # Distance between encods
CLEAN_TIMEOUT = 5

FACES_DIR = "faces/"    # Directory with faces encods
FACE_BORDER_PADDING = 0.05  # Padding for png preview for each encoding

g_SwitchThread = Thread()

# load all saved faces and return container with them
def load_faces() -> dict[str, list[FaceData]]:
    known_faces = {}
    if not os.path.isdir(FACES_DIR):
        os.mkdir(FACES_DIR)
    # initialize known mordas
    for face in os.listdir(FACES_DIR):
        face_dir = FACES_DIR + face
        if not os.path.isdir(face_dir):
            continue
        for data in os.listdir(face_dir):
            if data.endswith(".dat"):
                if face not in known_faces:
                    known_faces[face] = []
                dat_file = open(face_dir + "/" + data, "rb")
                img_file_name = data.split(".")[0] + ".png"
                known_faces[face].append(
                    FaceData(pickle.load(dat_file), str(data), str(img_file_name))
                )
                dat_file.close()

    return known_faces


# return img with face from face location on current frame
def make_face_preview(face_loc, frame: MatLike) -> MatLike:
    # make preview
    top, right, bottom, left = face_loc
    img = frame[
        int(top * (1 - FACE_BORDER_PADDING)) : int(bottom * (1 + FACE_BORDER_PADDING)),
        int(left * (1 - FACE_BORDER_PADDING)) : int(right * (1 + FACE_BORDER_PADDING)),
    ]
    return img


# save this face encoding called by 'name' and represented by 'img'. Returns saved face data
def save_face(
    face_enc, name, img
) -> tuple[bool, FaceData]:  # count = 0 # how much face datas we have
    if not SAVE_FLAG:
        return False, FaceData()
    dir = FACES_DIR + name
    id = uuid.uuid4().hex
    file = id + ".dat"
    filePath = dir + "/" + file
    imgfile = id + ".png"
    imgPath = dir + "/" + imgfile

    Path(dir).mkdir(exist_ok=True)
    dat_file = open(filePath, "wb")
    pickle.dump(face_enc, dat_file)
    cv2.imwrite(imgPath, img)

    dat_file.close()
    return FaceData(face_enc, file, imgfile)


# remove that face encoding from file system and virtual memory
# NEEDS REFACTOR
def remove_encoding(known_faces: dict[str, list[FaceData]], name: str, data: FaceData) -> None:
    known_faces[name].remove(data)
    file = FACES_DIR + name + "/" + data.get_filename()
    os.remove(file)
    file = FACES_DIR + name + "/" + data.get_picname()
    os.remove(file)


# get all encodings for curr face and return list of it in face_recognition format
def get_face_encods_list(known_faces: dict[str, list[FaceData]], name: str) -> list[MatLike]:
    enc_list = []
    for data in known_faces[name]:
        enc_list.append(data.get_enc())

    return enc_list

# remove all similar encods for all faces that is closer than max_dist and return count of removed encods
def remove_similar_encods(known_faces: dict[str, list[FaceData]], max_dist: float) -> int:
    total_removed = 0
    for name in known_faces:
        idxCurrFace = 0
        while idxCurrFace < len(known_faces[name]):
            facesData: list[MatLike] = get_face_encods_list(known_faces, name)

            currFaceData = known_faces[name][idxCurrFace].get_enc()
            dists = face_recognition.face_distance(facesData, currFaceData)

            nRemoved = 0
            idxDist = 0
            while idxDist < len(dists):
                if idxDist == idxCurrFace:
                    idxDist += 1
                    continue
                if dists[idxDist] < max_dist:
                    remove_encoding(
                        known_faces, name, known_faces[name][idxDist - nRemoved]
                    )
                    nRemoved += 1
                    total_removed += 1
                idxDist += 1

            idxCurrFace += 1

    return total_removed


# find that face_enc in known_faces and put its name in face_names. If doesnt found - return false,default_name
def recognize_face(
    face_enc, known_faces: dict[str, list[FaceData]], default_name, face_names
) -> tuple[bool, str]:
    faceInd = 0  # index for name
    found = False
    # lets go through encodinds for current face
    for face_enc_data in known_faces.values():
        face_enc_list = []
        for facedata in face_enc_data:
            face_enc_list.append(facedata.get_enc())
        match = face_recognition.compare_faces(
            face_enc_list, face_enc, tolerance=RECOG_TOLERANCE
        )  # find how current face matches our faces from the storage (euqlid distance from current face metrics to storage faces metrics)
        encIndex = 0  # index for encoding
        while encIndex < len(match):
            if match[encIndex]:
                default_name = list(known_faces.keys())[faceInd]
                face_names.append(default_name)
                found = True
                break
            encIndex += 1
        # didnt found -> lets check the next face and its encodings
        if found:
            break
        faceInd += 1

    if not found:
        face_names.append(default_name)
    return found, default_name

def main() -> None:
    global g_SwitchThread
    # load known faces
    known_faces = load_faces()

    # Get a reference to webcam
    video_capture = cv2.VideoCapture(0)
    #video_capture = cv2.VideoCapture('videos\epifan_crowd.mp4')

    # Initialize variables
    face_locations = []

    while True:
        # start cleaning same encodings
        bCleanStarted = False # flag about starting thread for cleaning
        if SAVE_FLAG: # if we dont save any encods then we dont need to clean
            clean_thread = Thread(target=remove_similar_encods, args=[known_faces, SIMILAR_FACES_DIST]) # thread for cleaning. Starts once at CLEAN_TIMEOUT seconds
            if not g_SwitchThread.is_alive():
                g_SwitchThread = Thread(target=lambda: (time.sleep(CLEAN_TIMEOUT)), args=[]) # timeout thread. If its alive then we shouldnt start cleaning
                g_SwitchThread.start()
                clean_thread.start()
                bCleanStarted = True
        
        # Grab a single frame of video
        ret, frame = video_capture.read()
        ret, frame = video_capture.read()
        ret, frame = video_capture.read()
        ret, frame = video_capture.read()
        ret, frame = video_capture.read()
        ret, frame = video_capture.read()
        if not ret:
            continue

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame: MatLike = cv2.resize(frame, (0, 0), fx=SCALE_KOEFF, fy=SCALE_KOEFF)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # container for faces names (we have faces encodings and names loaded in the same order)
        face_names = []

        # lets find faces on the frame
        for face in zip(face_locations, face_encodings):
            face_loc = face[0]  # all faces locations
            face_enc = face[1]  # all located faces encodings
            name = "UNKNOWN" # default name
            if SAVE_FLAG:
                name += str(len(known_faces) + 1)
            found, name = recognize_face(
                face_enc, known_faces, name, face_names
            )  # recognize any faces in frame
            
            if bCleanStarted: # if its started then we must wait until it ends
                clean_thread.join()
                
            if found:
                if len(known_faces[name]) < FACES_ENCODS:
                    success, new_face = save_face(
                        face_enc, name, make_face_preview(face_loc, frame)
                    )
                    if success:
                        known_faces[name].append(new_face)
                continue

            success, new_face = save_face(face_enc, name, make_face_preview(face_loc, frame))
            if success:
                known_faces[name] = [new_face]

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(
                frame,
                name,
                (left + 5, bottom - 5),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.4,
                color=(0, 0, 255),
            )

        # Display the resulting image
        cv2.imshow("Video", frame)

        # Hit 'q' on the keyboard to quit!
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord("q"):
            g_Cleaning = False
            break
        # hit 'r' to reload faces (when rename etc)
        if pressed_key == ord("r"):
            known_faces = load_faces()
            tkinter.messagebox.showinfo("Info", "Reloaded faces")
        if pressed_key == ord("u"):
            th = Thread(target=(lambda faces_container: tkinter.messagebox.showinfo("Info", "Removed similar face encods: " + str(remove_similar_encods(faces_container, SIMILAR_FACES_DIST)))), args=[known_faces])
            th.start()

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


main()
