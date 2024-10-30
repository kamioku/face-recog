from cv2.typing import MatLike

class FaceData:
    def __init__(self, enc, name):
        self.__enc = enc
        self.__file = name
    
    def get_enc(self):
        return self.__enc

    def get_filename(self):
        return self.__file


    __enc: MatLike
    __file: str
    
