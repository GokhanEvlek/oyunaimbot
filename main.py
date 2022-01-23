import torch
from PIL import ImageGrab
import cv2 as cv
import numpy as np
import os
import pyautogui
from pynput.mouse import Button, Controller
import win32api, win32con
import ctypes
import pydirectinput
import time
#model=torch.hub.load('ultralytics/yolov5','custom','C:\\Users\\Gökhan\\PycharmProjects\\oyunaimbot\\yolov5\\aimbotairlik.pt',force_reload=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
classes=model.names
mouse=Controller()
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions
def MouseMoveTo(x, y):
    x = 1 + int(x * 65536./1920.)#1920 width of your desktop
    y = 1 + int(y * 65536./1080.)#1080 height of your desktop
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi =  MouseInput(x,y,0, (0x0001 | 0x8000), 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(0), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
while True:
    img = ImageGrab.grab()  # take a screenshot

    img=np.array(img)
    img=cv.cvtColor(img,cv.COLOR_RGB2BGR)
    cv.imwrite("image1.png", img)

    #img=cv.imread(img)
    #img= cv.imread(img)
    img=[img]
    results = model(img)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    n = len(labels)

    img = cv.imread("image1.png")
    #print(labels)
    x_shape = img.shape[1]
    y_shape = img.shape[0]
    for i in range(0,n):
        #labels1=results.xyxyn[0][:,5:].cpu()
        #label=labels1.numpy()
        #if label[0]==0:
        row = cord[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            sınıf=classes[int(labels[i])]
            if sınıf=="person":


                img=cv.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
                img=cv.putText(img, sınıf, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
                ourcordsx=(x1+x2)/2
                ourcordsy=y1
                #pyautogui.moveTo(ourcordsx,ourcordsy,0)
                #MouseMoveTo(ourcordsx,ourcordsy)
                #mouse.position=(ourcordsx,ourcordsy)
                #mouse.click(Button.left,1)
                #pydirectinput.move(int(ourcordsx),int(ourcordsy))
                #pydirectinput.keyDown('w')
                #time.sleep(1)
                #pydirectinput.keyUp('w')
                convertedX = 65536 * ourcordsx // x_shape + 1
                convertedY = 65536 * ourcordsy // y_shape + 1
                MouseMoveTo(int(ourcordsx),int(ourcordsy))
                #pydirectinput.keyDown('w')
                #time.sleep(1)
                #pydirectinput.keyUp('w')
                
                
                print("hareket fn'u çalıştı")
                #ctypes.windll.user32.mouse_event(ev, ctypes.c_long(convertedX), ctypes.c_long(convertedY), dwData, 0)
                #win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, int(convertedX),int(convertedY),0,0)
                pydirectinput.click()
        else:
            continue
    #cv.imshow("adam", img)
    #k=cv.waitKey(1)
    os.remove("image1.png")
    #if k==27:
        #cv.destroyAllWindows()
        #break
