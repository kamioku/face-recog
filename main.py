# import libraries
import cv2
import face_recognition
import os
import pickle
import uuid
import tkinter.messagebox

from pathlib import Path

SCALE_KOEFF = 0.5

FACES_DIR = "faces/"
FACE_BORDER_PADDING = 0.05

def load_faces():
    known_faces = {}
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
                known_faces[face].append(pickle.load(dat_file))
                dat_file.close()
    
    return known_faces

def make_face_preview(face_loc, frame):
    # make preview
    top, right, bottom, left = face_loc
    img = frame[int(top*(1-FACE_BORDER_PADDING)):int(bottom*(1+FACE_BORDER_PADDING)), int(left*(1-FACE_BORDER_PADDING)):int(right*(1+FACE_BORDER_PADDING))]
    #cv2.imwrite(FACES_DIR+name+".png", img)
    return img

def save_face(face_enc, name, img):
    count = 0 # how much face datas we have
    dir = FACES_DIR+name
    file = dir+"/"+uuid.uuid4().hex+"_"+str(count)+".dat"
    imgfile = dir+"/"+name+".png"
    
    while(os.path.isfile(file)):
        count += 1
        file = dir+"/"+name+count+".dat"
        
    Path(dir).mkdir(exist_ok=True)
    dat_file = open(file, 'wb')
    pickle.dump(face_enc, dat_file)
    cv2.imwrite(imgfile, img)
    
    dat_file.close()

def main():
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
        frame = cv2.resize(frame, (0,0), fx=SCALE_KOEFF, fy=SCALE_KOEFF)
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
            
            faceInd = 0 # index for name
            found = False
            
            # lets go through encodinds for current face
            for face_enc_data in known_faces.values():
                match = face_recognition.compare_faces(list(face_enc_data), face_enc) # find how current face matches our faces from the storage (euqlid distance from current face metrics to storage faces metrics)
                name = "UNKNOWN" + str(len(known_faces)+1) # default name
                
                encIndex = 0 # index for encoding
                while encIndex < len(match):
                    if match[encIndex]:
                        name = list(known_faces.keys())[faceInd]
                        face_names.append(name)
                        found = True
                        break
                    encIndex += 1
                # didnt found -> lets check the next face and its encodings
                faceInd += 1
                    
            if found:
                continue
            
            save_face(face_enc, name, make_face_preview(face_loc, frame))
            known_faces[name] = [face_enc]
            
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations,face_names):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left+5, bottom-5), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0,0,255))
            
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