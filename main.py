# import libraries
import cv2
import face_recognition
import os
import pickle
import uuid
import tkinter.messagebox

from pathlib import Path

from modules.face_data import FaceData

SCALE_KOEFF = 0.5

FACES_ENCODS = 5

FACES_DIR = "faces/"
FACE_BORDER_PADDING = 0.05

def load_faces() -> dict[str, list[FaceData]]:
    known_faces = {}
    if not os.path.isdir(FACES_DIR):
        os.mkdir(FACES_DIR)
    # initialize known mordas
    for face in os.listdir(FACES_DIR):
        face_dir = FACES_DIR+face
        if(not os.path.isdir(face_dir)):
            continue
        for data in os.listdir(face_dir):
            if data.endswith(".dat"):
                if face not in known_faces:
                    known_faces[face] = []
                dat_file = open(face_dir+"/"+data, 'rb')
                known_faces[face].append(FaceData(pickle.load(dat_file), data))
                dat_file.close()
    
    return known_faces

def make_face_preview(face_loc, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    # make preview
    top, right, bottom, left = face_loc
    img = frame[int(top*(1-FACE_BORDER_PADDING)):int(bottom*(1+FACE_BORDER_PADDING)), int(left*(1-FACE_BORDER_PADDING)):int(right*(1+FACE_BORDER_PADDING))]
    #cv2.imwrite(FACES_DIR+name+".png", img)
    return img
def save_face(face_enc, name, img) -> FaceData: #count = 0 # how much face datas we have
    dir = FACES_DIR+name
    id = uuid.uuid4().hex
    file = dir+"/"+id+".dat"
    imgfile = dir+"/"+id+".png"
    
    # if for some reason we have same id in that folder
    while(os.path.isfile(file)):
        id = uuid.uuid4().hex
        file = dir+"/"+id+".dat"
        imgfile = dir+"/"+id+".png"
        
    Path(dir).mkdir(exist_ok=True)
    dat_file = open(file, 'wb')
    pickle.dump(face_enc, dat_file)
    cv2.imwrite(imgfile, img)
    
    dat_file.close()
    return FaceData(face_enc, id)

def remove_encoding(known_faces: dict[str, list[FaceData]], name, data:FaceData) -> None:
    known_faces[name].remove(data)

    file = FACES_DIR+name+'/'+data.get_filename()
    os.remove(file)

def get_face_encods_list(known_faces: dict[str, list[FaceData]], name) -> list[FaceData]:
    enc_list = []
    for data in known_faces[name]:
        enc_list.append(data.get_enc())

    return enc_list

def remove_similar_encods(known_faces, name, max_dist) -> None:
    dists = []

    for face in get_face_encods_list(known_faces, name):
        dists = face_recognition.face_distance(known_faces[name][1:], face)

    idxDist = 0
    while idxDist < len(dists):
        if dists[idxDist] < max_dist:
            remove_encoding(known_faces, name, dists[idxDist])

    
def recognize_face(face_enc, known_faces: dict[str, list[FaceData]], default_name, face_names) -> tuple[bool, str]:
    faceInd = 0 # index for name
    found = False
    # lets go through encodinds for current face
    for face_enc_data in known_faces.values():
        face_enc_list = []
        for facedata in face_enc_data:
            face_enc_list.append(facedata.get_enc())
        match = face_recognition.compare_faces(face_enc_list, face_enc, tolerance=0.6) # find how current face matches our faces from the storage (euqlid distance from current face metrics to storage faces metrics)    
        encIndex = 0 # index for encoding
        while encIndex < len(match):
            if match[encIndex]:
                default_name = list(known_faces.keys())[faceInd]
                face_names.append(default_name)
                found = True
                break
            encIndex += 1
        # didnt found -> lets check the next face and its encodings
        faceInd += 1
        
    return found, default_name

def main() -> None:
    # load known faces
    known_faces = load_faces()
    
    # Get a reference to webcam
    video_capture = cv2.VideoCapture(0)
    
    # Initialize variables
    face_locations = []
    
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            continue
            
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame: cv2.typing.MatLike = cv2.resize(frame, (0,0), fx=SCALE_KOEFF, fy=SCALE_KOEFF)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces in the current frame of video    
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # container for faces names (we have faces encodings and names loaded in the same order)
        face_names = []
        
        # lets find faces on the frame
        for face in zip(face_locations,face_encodings):
            face_loc = face[0] # all faces locations
            face_enc = face[1] # all located faces encodings
            
            name = "UNKNOWN" + str(len(known_faces)+1) # default name
            found, name = recognize_face(face_enc, known_faces, name, face_names) # recognize any faces in frame
            
            if found:
                if len(known_faces[name]) < FACES_ENCODS:
                    new_face = save_face(face_enc, name, make_face_preview(face_loc, frame))        
                    known_faces[name].append(new_face)
                continue
            
            new_face = save_face(face_enc, name, make_face_preview(face_loc, frame))
            known_faces[name] = [new_face]
            
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations,face_names):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left+5, bottom-5), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.4, color=(0,0,255))
            
        # Display the resulting image
        cv2.imshow('Video', frame)
        
        # Hit 'q' on the keyboard to quit!
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            break
        # hit 'r' to reload faces (when rename etc)
        if pressed_key == ord('r'):
            known_faces = load_faces()
            tkinter.messagebox.showinfo('Info','Reloaded faces')
    
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    
main()
