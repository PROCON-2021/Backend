import serial
import re
from multiprocessing import Array, Value

#demo
"""
def run(flag: Value, ch1: Array, ch2: Array, ch3: Array):
    array = [""] * 4000
    while flag.value != -1:
        i = 0
        while flag.value == 0:
            array[i] = "1,2," + str(i)
            i += 1
            if i == 4000:
                i -= 4000
        
        if flag.value == 1:
            i += 500
            for j in range(0, 3500):
                buf = array[i % 4000].split(',')
                ch1[j] = int(buf[0])
                ch2[j] = int(buf[1])
                ch3[j] = int(buf[2])
                i += 1
            flag.value = 0
            
    flag.value = 0

"""
#serial
def run(flag: Value, ch1: Array, ch2: Array, ch3: Array):
    ser = serial.Serial("COM3", 38400, timeout = 1)
    array = ["500,500,500,500"] * 4000
    while flag.value != -1:
        i = 0
        while flag.value == 0:
            serstr = str(ser.readline())
            strs = re.findall("[0-9]{1,4},[0-9]{1,4},[0-9]{1,4},[0-9]{1,4}", serstr)
            if len(strs) != 0:
                array[i] = strs[0]
                i += 1
                if i == 4000:
                    i -= 4000
        
        if flag.value == 1:
            i += 500
            for j in range(0, 3500):
                buf = array[i % 4000].split(',')
                ch1[j] = int(buf[0])
                ch2[j] = int(buf[1])
                ch3[j] = int(buf[2])
                i += 1
            flag.value = 0

    ser.close()
    flag.value = 0