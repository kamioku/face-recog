from cv2.typing import MatLike


class FaceData:
    
    def __init__(self, enc: MatLike = [], datName: str = "", picName: str = ""):
        self.__enc = enc
        self.__file = datName
        self.__pic = picName

    def get_enc(self) -> MatLike:
        return self.__enc

    def get_filename(self) -> str:
        return self.__file

    def get_picname(self) -> str:
        return self.__pic

    def set_picname(self, name: str):
        self.__pic = name

    __enc: MatLike
    __file: str
    __pic: str
