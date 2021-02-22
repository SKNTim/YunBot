import cv2
import numpy as np


class PerspectiveTransform(object):
    """透視轉換

    Args:
        image_shape: (tuple)影像大小(w, h)
        src_vector: (list)透視矩陣
    """
    def __init__(self, image_shape, src_vector=[[110,100],[310,100],[420,200],[0,200]]):
        self.image_shape = image_shape
        width, height = image_shape
        src = np.float32(src_vector)
        dst = np.float32([[0,0],[width,0],[width,height],[0,height]])

        self.transform_matrix = cv2.getPerspectiveTransform(src, dst)
        self.revert_matrix = cv2.getPerspectiveTransform(dst, src)

    def transform(self, img):
        """ 透視轉換 """
        return self._warp_perspective(img, self.transform_matrix)

    def revert(self, img):
        """ 透視還原 """
        return self._warp_perspective(img, self.revert_matrix)

    def _warp_perspective(self, img, matrix):
        return cv2.warpPerspective(img, matrix, self.image_shape,
                                   flags=cv2.INTER_LINEAR)
