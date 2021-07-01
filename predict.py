from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
#from alg import Data_Analysis
import numpy as np
from log import get_lgger

class Data_Analysis:
    def __init__(self):
        self.IMAGE_Size=150

    def clean(self, images):
        # To load the images and convert into numpy array format.
        #self.input_images = images
        try:
            array_values = cv2.imread(images)  # reading the image and converting it into array
            print(array_values.shape)
            array_values = cv2.resize(array_values, (self.IMAGE_Size, self.IMAGE_Size))
            print(array_values.shape)
            return array_values
        except Exception as e:
            t=get_lgger('ERROR')
            t.error(str(e))
            print('Exception occured ,{}'.format(str(e)))
            return None


    def convert(self, array_images):
        # To apply convert like dividing with 255
        try:
            self.array_images = np.array(array_images) / 255
            return self.array_images
        except Exception as e:
            print('Exception Occured ,{}'.format(str(e)))
            return None

def predit(inputimage):
    algorithm = load_model('alg.h5')
    try:
        print('input image is ',inputimage)
        #img=cv2.imread(inputimage)
        analysis=Data_Analysis()
        image_array=analysis.clean(inputimage)
        print(' after cleaning. ')
        img=[]
        img.append(image_array)
        np_image=analysis.convert(img)
        print('after image convert.',np_image.shape)
        result=algorithm.predict(np_image)
        print('during predict.')
        return np.argmax(result,axis=1)
    except Exception as e:
        return "Some exception occured. {}".format(str(e))


