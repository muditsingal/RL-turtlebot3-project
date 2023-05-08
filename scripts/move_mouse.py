#! python3
import pyautogui, sys
import time
print('Press Ctrl-C to quit.')
try:
    while True:
        pyautogui.moveTo(200,400)
        time.sleep(100)
        pyautogui.moveTo(400,200)
        time.sleep(100)
except KeyboardInterrupt:
    print('\n')