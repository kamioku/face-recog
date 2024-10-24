# import libraries
import cv2
import face_recognition
import os
import pickle

from pathlib import Path

SCALE_KOEFF = 0.5

FACES_DIR = "faces/"
FACE_BORDER_PADDING = 0.05

# def load_faces():
#     known_faces = {}
#     # initialize known mordas
#     for img in os.listdir(FACES_DIR):
#         name = img.rstrip(".png")
#         image = face_recognition.load_image_file(FACES_DIR + img)
#         for face_encoding in face_recognition.face_encodings(image):
#             known_faces[name] = face_encoding
            
#     return known_faces
def load_faces():
    known_faces = {}
    # initialize known mordas
    for face in os.listdir(FACES_DIR):
        face_dir = FACES_DIR+face
        if(not os.path.isdir(face_dir)):
            continue
        for data in os.listdir(face_dir):
            if data.endswith(".dat"):
                dat_file = open(face_dir+"/"+data, 'rb')
                known_faces[face] = pickle.load(dat_file)
                dat_file.close()
    
    return known_faces

def make_face_preview(face_loc, frame):
    # make preview
    top, right, bottom, left = face_loc
    img = frame[int(top*(1-FACE_BORDER_PADDING)):int(bottom*(1+FACE_BORDER_PADDING)), int(left*(1-FACE_BORDER_PADDING)):int(right*(1+FACE_BORDER_PADDING))]
    #cv2.imwrite(FACES_DIR+name+".png", img)
    return img

def save_face(face_enc, name, img):
    dir = FACES_DIR+name
    file = dir+"/"+name+".dat"
    imgfile = dir+"/"+name+".png"
    #os.mkdir(dir, exist_ok=True)
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
        
            
        # Find all the faces in the current frame of video
            
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_names = []
        
        for face in zip(face_locations,face_encodings):
            face_loc = face[0]
            face_enc = face[1]
            match = face_recognition.compare_faces(list(known_faces.values()), face_enc, tolerance=0.5)
            name = "UNKNOWN" + str(len(known_faces)+1)
            
            iter = 0
            found = False
            while iter < len(match):
                if match[iter]:
                    name = list(known_faces.keys())[iter]
                    face_names.append(name)
                    found = True
                    break
                iter+=1
            if found:
                continue
            
            # top, right, bottom, left = face_loc
            # cv2.imwrite(FACES_DIR+name+".png", frame[int(top*(1-FACE_BORDER_PADDING)):int(bottom*(1+FACE_BORDER_PADDING)), int(left*(1-FACE_BORDER_PADDING)):int(right*(1+FACE_BORDER_PADDING))])
            # known_faces[name] = face_enc
            # face_names.append(name)
            save_face(face_enc, name, make_face_preview(face_loc, frame))
            known_faces[name] = face_enc
            
        # Display the results
            
        for (top, right, bottom, left), name in zip(face_locations,face_names):
                    
            # Draw a box around the face
                    
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left+5, bottom-5), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0,0,255))
            
            
        # Display the resulting image
            
        cv2.imshow('Video', frame)
        
            
        # Hit 'q' on the keyboard to quit!
            
        if cv2.waitKey(1) & 0xFF == ord('q'):    
            break
        if cv2.waitKey(1) & 0xFF == ord('r'):
            known_faces = load_faces()
    
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    
main()