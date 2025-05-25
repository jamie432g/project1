import cv2
import numpy as np

class ViewTransformer:
    def __init__(self):
        pass
    def get_homography(self,source,target):
        #compute homography from source points to target points
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        h_matrix,_ = cv2.findHomography(source,target)
        if h_matrix is None:
            print ("Error, homography matrix could not be found")
            return None
        else:
            return h_matrix
    def homography_transform(self,source,target,points):
        #homography applied to given points
        if source.shape != target.shape:
            print("Shape mismatch between source and target")
        if points.size == 0:
            return points
        h_matrix = self.get_homography(source,target)
        points = points.reshape(-1,1,2).astype(np.float32)
        points = cv2.perspectiveTransform(points, h_matrix)
        return points.reshape(-1,2).astype(np.float32)

