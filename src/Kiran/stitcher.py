import glob
import cv2
import os
from src.Kiran.Imp_funcs import Funcs
class PanaromaStitcher():
    def __init__(self):
        pass
        

    def make_panaroma_for_images_in(self,path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))
        self.imageSet = [cv2.imread(each) for each in all_images]
        self.images = [cv2.resize(each,(480,320)) for each in self.imageSet ]
        self.images = [Funcs.change_xy_to_cylindrical(each) for each in self.images]
        self.center = int(len(self.images)/2)
        self.say_hi()        
        # Return Final panaroma
        stitched_image,H = Funcs.stitching_func(self.images, self.center)
        homography_matrix_list =H
        
        return stitched_image, homography_matrix_list

    def say_hi(self):
        print('Hi From Kiran')
    
    