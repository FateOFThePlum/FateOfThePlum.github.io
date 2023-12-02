#from picamera2 import Picamera2
#from libcamera import controls
import easygui
from datetime import datetime
import cv2
import time
import numpy as np
from pyzbar import pyzbar
import math

class measure_class(object):
    pass
measurement = measure_class()
measurement.height = 9
measurement.width = 5
global average_pixel_to_inch_ratio
global img
global frame
global ksize
global qr_codes_detected





#Settings For Logan
Source = 'file' #'file' or ''
Auto_Exposure = False
Debug_mode = True
s22_Focal_Length = 5533.769270061575 #This Was Calculated By https://github.com/Asadullah-Dal17/QR-detection-and-Distance-Estimation/blob/main/main.py
Line_Thickness = 20 #This Controls the thickness of the lines that surround the qrcode(s) and the reflective tape 
actual_inch_length = 2  # Replace with the actual inch length of the QR code 
                        #Make Sure The Actual Length/Perimiter Accounts for the Potential Scaling  

#Setting Up Variables
#Just Sets up variable so it doesn't throw errors at me 
perimeter_sum = 0 
qr_code_count = 0

def distanceFinder(focalLength, knownWidth, widthInImage):
    # Source -=> https://github.com/Asadullah-Dal17/QR-detection-and-Distance-Estimation/blob/main/main.py
    '''
    This function basically estimate the distance, it takes the three arguments: focallength, knownwidth, widthInImage
    :param1 focalLength: focal length found through another function .
    param2 knownWidth : it is the width of object in the real world.
    param3 width of object: the width of object in the image .
    :returns the distance:
    '''
    
    distance = ((knownWidth * focalLength) / widthInImage)
    return distance


def printer (*args):
    #This Function is to make it so I can Turn Deguging On and Off
    #It is controlled by the Debug_mode Boolean Variable 
    if Debug_mode == True:
        print(*args)
        return
    else:
        return

def insize(size): 
    # This Function is for The Rough Measurement of the reflective Tape/Marker

    tolerance = 0.075 # + or - tolerance 
    size -= 1
    if size < tolerance and size > tolerance*-1:
        return True
    else:
        return False

def sizefinder (input_height, input_width):
    ratio = 1/input_height # Reflective Tape is 0.977 At Start Of Role
    real_height = input_height * ratio
    real_width = input_width * ratio
    return  round(real_height, 4), round(real_width, 4)

def sizefinderV2 (h, w, pixel_ratio):
    input_height = h / pixel_ratio
    ratio = 1/input_height # Reflective Tape is 0.977 At Start Of Roll
    pixel_ratio = ratio * pixel_ratio
    real_height = h / pixel_ratio
    real_width = w / pixel_ratio
    return  round(real_height, 4), round(real_width, 4)

def mask_func(frame):
    # Object detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower = np.array([85, 165, 0])
        upper = np.array([120, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 1) #Blurs 
        
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                objects_contours.append(cnt)
        return gray_result, contours, objects_contours

def BigPortionV1 (img, frame):
    printer("Detecting QR Codes...")
    perimeter_sum = 0
    qr_code_count = 0
    qr_codes_detected = False
    object_height, object_width = 0, 0
    # QR code detection
    qr_codes = pyzbar.decode(frame)
    #frame = cv2.resize(frame, (0, 0), fx = 0.25, fy = 0.25)
    #img = cv2.resize(img, (0, 0), fx = 0.25, fy = 0.25)
    maskImg = img


    for qr_code in qr_codes:
        qr_data = qr_code.data.decode('utf-8')
        printer("QR Data:", qr_data)
        id_tag = qr_data

        # Calculate perimeter using QR code polygon points
        qr_polygon = qr_code.polygon
        qr_perimeter = 0
        for i in range(len(qr_polygon)):
            qr_perimeter += np.linalg.norm(np.array(qr_polygon[i]) - np.array(qr_polygon[(i + 1) % len(qr_polygon)]))


        perimeter_sum += qr_perimeter
        qr_code_count += 1

        # Draw QR code bounding box using polygon
        for j in range(len(qr_polygon)):
            cv2.line(img, qr_polygon[j], qr_polygon[(j + 1) % len(qr_polygon)], (0, 255, 0), Line_Thickness)
    
    if qr_code_count == 0:
        printer('No Qr-Codes Found')
        
    if qr_code_count > 0:
        printer('QR Code Count: ', qr_code_count)
        qr_codes_detected = True
        
        # Calculate average pixel to inch ratio
        
        global actual_inch_length 

        average_pixel_to_inch_ratio = perimeter_sum / (qr_code_count * actual_inch_length)
        printer("Average Pixel to Inch Ratio:", average_pixel_to_inch_ratio)

        if Auto_Exposure == False:
            img, object_height, object_width, height_pixels = measurement_img_Auto_Exposure_False(maskImg, img, average_pixel_to_inch_ratio)
        elif Auto_Exposure == True:
            printer('Auto Exposure Feature Has Been Removed')
    try:
        distance_From_QR_Code = distanceFinder(s22_Focal_Length, 2, (perimeter_sum/qr_code_count))
        distance_From_Target = distanceFinder(s22_Focal_Length, 1, (height_pixels))
        printer('Distance From Qrcode: ', distance_From_QR_Code, 'Distance From Target: ', distance_From_Target)
    except ZeroDivisionError:
        #printer('Zero ERROR')
        pass
    return img, object_height, object_width, qr_codes_detected


def measurement_img_Auto_Exposure_False(maskImg, img, average_pixel_to_inch_ratio):
    global object_height, object_width, measurement

    gray_result, contours, objects_contours = mask_func(maskImg)
    cv2.imshow("Mask", cv2.resize(gray_result, (0, 0), fx = 0.25, fy = 0.25))
    if not objects_contours:
        printer('No Object Contures')
        return img, 0, 0, 0
    for cnt in objects_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        #(x, y), (w, h) #This Is The Format
        object_width = w / average_pixel_to_inch_ratio
        object_height = h / average_pixel_to_inch_ratio
        measurement.height = object_height
        measurement.width = object_width
        if insize(object_height) == True:
            object_height, object_width = sizefinder(object_height, object_width)
            printer("Exposure Status == ", Auto_Exposure,' Width:', object_width,' Height:', object_height)
            
            #object_height, object_width = sizefinderV2(w, h, average_pixel_to_inch_ratio)
            #printer("V2","Exposure:", 'exposure_off',' Width:', object_width,' Height:', object_height)
            
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), Line_Thickness)
            cv2.putText(img, "Width {} in".format(round(object_width, 1)), (int(x), int(y + h + 115)), cv2.FONT_HERSHEY_PLAIN, Line_Thickness/2, (120, 170, 0), Line_Thickness)
            cv2.putText(img, "Height {} in".format(round(object_height, 1)), (int(x), int(y + h + 230)), cv2.FONT_HERSHEY_PLAIN, Line_Thickness/2, (120, 170, 0), Line_Thickness)
            printer('Height: ', object_height, 'Width: ', object_width)
            return img, object_height, object_width, h
        else: 
            printer('Object Not Within Size')
            printer('Height: ', object_height, 'Width: ', object_width)
            return img, 0, 0, 0

def auto_median_blur(og_img):
    global qr_codes_detected
    ksize = 0
    while qr_codes_detected == False:
        
        
        if ksize == 0:
            printer('Starting Process...')
            printer('Ksize: ', ksize)
            img = og_img.copy()
        elif ksize % 2 == 1:
            printer('Ksize: ', ((ksize/2)+0.5))
            img = cv2.medianBlur(og_img, ksize)
        frame = img.copy()
        if ksize == 0 or ksize % 2 == 1:
            img, object_height, object_width, qr_codes_detected = BigPortionV1(img, frame)

        if ksize >= 15:
            qr_codes_detected = True

        ksize += 1
        if Debug_mode == True:
            cv2.imshow('Final Img', cv2.resize(img, (0, 0), fx = 0.125, fy = 0.125))


def app_port(og_img):
    global qr_codes_detected
    printer('You Are Useing')
    qr_codes_detected = False
    while qr_codes_detected == False:
        auto_median_blur(og_img)

if __name__ == "__main__":
    printer('Debug Mode == ', Debug_mode) #Just To Tell Me if Debug Mode Is Turned On or Not

    while True:
        printer('Running From Main Program')
        #This Next Chunk is to pick a file from the computer
        if Source == 'file':
            try: 
                File_Name = easygui.fileopenbox()
                if Debug_mode == True:
                    printer('Selected Path', File_Name)
                og_img = cv2.imread(File_Name,1)
                img = og_img.copy()
                #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Correct For camera Format
            except AttributeError:
                printer('\033[1m','\n\n', ('Error Reading File'.center(25,)), '\033[1m')
                printer('\033[1m', ('Please Use a .png or .jpg'.center(25,)), '\033[1m')
                printer('\033[1m', ("Don't Close File Explorer".center(25,)), '\033[1m')
                printer('\033[1m', ('Stopping Program...'.center(25,)), '\n', '\033[1m')
                break
        elif Source == 'camera' or Source == 'cam' or Source == 'live':
            #I Havent Programmed This Part Yet :(
            pass
      
        printer('2')
        qr_codes_detected = False
        while qr_codes_detected == False:
            auto_median_blur()

    cv2.destroyAllWindows()
    printer('\033[1m','\n\n', ('Program Stopped'.center(25,)), '\033[1m')
    printer('\033[1m', ('Cause: Exited Loop'.center(25,)), '\n\n', '\033[1m')