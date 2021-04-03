"""
Created on Thursday June 11 16:38:42 2020
@author: Utkarsh Deshmukh
"""
import cv2
import numpy as np

from Airlight import Airlight
from BoundCon import BoundCon
from CalTransmission import CalTransmission
from removeHaze import removeHaze
import os

def dehaze_image(file_url,base_url):
    HazeImg = cv2.imread(file_url)

    # Resize image
    '''
    Channels = cv2.split(HazeImg)
    rows, cols = Channels[0].shape
    HazeImg = cv2.resize(HazeImg, (int(0.4 * cols), int(0.4 * rows)))
    '''

    # Estimate Airlight
    windowSze = 15
    AirlightMethod = 'fast'
    A = Airlight(HazeImg, AirlightMethod, windowSze)

    # Calculate Boundary Constraints
    windowSze = 3
    C0 = 20         # Default value = 20 (as recommended in the paper)
    C1 = 300        # Default value = 300 (as recommended in the paper)
    Transmission = BoundCon(HazeImg, A, C0, C1, windowSze)                  #   Computing the Transmission using equation (7) in the paper

    # Refine estimate of transmission
    regularize_lambda = 1       # Default value = 1 (as recommended in the paper) --> Regularization parameter, the more this  value, the closer to the original patch wise transmission
    sigma = 0.5
    Transmission = CalTransmission(HazeImg, Transmission, regularize_lambda, sigma)     # Using contextual information

    # Perform DeHazing
    HazeCorrectedImg = removeHaze(HazeImg, Transmission, A, 0.85)

    result_url = base_url + "/media/images"
    file_name = os.path.basename(img_A_path)
    img_name = result_folder+'/' + file_name

    imwrite(img_name, HazeCorrectedImg)

    #cv2.imwrite('outputImages/result.jpg', HazeCorrectedImg)
