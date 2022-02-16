import torch
from PIL import ImageGrab
import cv2 as cv
import numpy as np
import os
import win32api, win32con, win32gui
import ctypes
import time
#model=torch.hub.load('ultralytics/yolov5','custom','C:\\Users\\Gökhan\\PycharmProjects\\oyunaimbot\\yolov5\\aimbotairlik.pt',force_reload=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
classes=model.names
size_scale = 3
while True:
    img = ImageGrab.grab(bbox =(559, 254, 1358, 851)) # take a screenshot
    #img = ImageGrab.grab(bbox =(36, 25, 1889, 1037))
    #hwnd = win32gui.FindWindow(None, 'Counter-Strike: Global Offensive')
    #hwnd = win32gui.FindWindow("UnrealWindow", None) # Fortnite
    #rect = win32gui.GetWindowRect(hwnd)
    #region = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]

    # Get image of screen
    #img = np.array(pyautogui.screenshot(region=region))
    #img_w, img_h = img.shape[1], img.shape[0]
    img=np.array(img)
    img=cv.cvtColor(img,cv.COLOR_RGB2BGR)
    cv.imwrite("image1.png", img)

    #img=cv.imread(img)
    #img= cv.imread(img)
    img=[img]
    results = model(img)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    n = len(labels)
    keys = Keys()
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
                x=int((x1+x2)/2-x_shape/2)
                y =int(y1+20 - y_shape/2)
                ourcordsx=(x1+x2)/2
                ourcordsy=y1
                #keys.directMouse(-1*int((x_shape/2)-ourcordsx), -1*int((y_shape/2)-ourcordsy))
                #keys.directMouse(buttons=keys.mouse_lb_press)
                #keys.directMouse(buttons=keys.mouse_lb_release)
                scale = 2.5
                #x = int(((x_shape/2)-ourcordsx)*scale)
                #y = int(((y_shape/2)-ourcordsy)*scale)
                x=int(x*scale)
                y=int(y*scale)
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)
                time.sleep(0.05)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
                time.sleep(0.1)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
                time.sleep(0.05)
                #pyautogui.moveTo(ourcordsx,ourcordsy,0)
                #MouseMoveTo(ourcordsx,ourcordsy)
                #mouse.position=(ourcordsx,ourcordsy)
                #mouse.click(Button.left,1)
                #pydirectinput.move(int(ourcordsx),int(ourcordsy))
                #pydirectinput.keyDown('w')
                #time.sleep(1)
                #pydirectinput.keyUp('w')
                #convertedX = 65536 * ourcordsx // x_shape + 1
                #convertedY = 65536 * ourcordsy // y_shape + 1
                #MouseMoveTo(int(ourcordsx),int(ourcordsy))
                #pydirectinput.keyDown('w')
                #time.sleep(1)
                #pydirectinput.keyUp('w')
                
                
                print("hareket fn'u çalıştı")
                #ctypes.windll.user32.mouse_event(ev, ctypes.c_long(convertedX), ctypes.c_long(convertedY), dwData, 0)
                #win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, int(convertedX),int(convertedY),0,0)
                #pydirectinput.click()
        else:
            continue
    cv.imshow("adam", img)
    k=cv.waitKey(1)
    os.remove("image1.png")
    if k==27:
        cv.destroyAllWindows()
        break


