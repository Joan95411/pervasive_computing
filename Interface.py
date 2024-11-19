from tkinter import *
from test import train_test
import random
import numpy as np
import time
import serial

root = Tk()
root.title("Tk Example")
root.minsize(200, 200)  # width, height
root.geometry("800x400+50+50")  # width x height + x + y

default_image = PhotoImage(file="Noone.pgm")
bed_image = PhotoImage(file="Bed.pgm")
desk_image = PhotoImage(file="Desk.pgm")
bathroom_image = PhotoImage(file="Bathroom.pgm")

ser = serial.Serial('COM3', 9600)
# Allow some time for the Arduino to initialize
time.sleep(2)
current_degree = 0

def manual():

    bed["state"] = "active"
    bathroom["state"] = "active"
    desk["state"] = "active"
    manualControl["state"] = "disabled"
    autoControl["state"] = "active"


def auto():

    bed["state"] = "disabled"
    bathroom["state"] = "disabled"
    desk["state"] = "disabled"
    manualControl["state"] = "active"
    autoControl["state"] = "disabled"


def pointToBed():
    direct = "bed"
    print("direction: ", direct)
    updateLabel(direct)
    degree = map_space_to_degree(direct)
    send_serial_data(degree)

def pointToDesk():
    direct = "desk"
    print("direction: ", direct)
    updateLabel(direct)
    degree = map_space_to_degree(direct)
    send_serial_data(degree)

def pointToBathroom():
    direct = "bathroom"
    print("direction: ", direct)
    updateLabel(direct)
    degree = map_space_to_degree(direct)
    send_serial_data(degree)

def send_serial_data(degree):
    global current_degree
    if current_degree != degree:
        turnOnFan()
        turn_degree = degree - current_degree
        data = str(abs(turn_degree)) + ('C' if turn_degree >= 0 else 'A')
        ser.write(data.encode())
        print("Sent Data:", data)
        # Update the current degree
        current_degree = degree




def turnOffFan():
    direct = "default"
    print("direction: ", direct)
    updateLabel(direct)
    off_data = str(0) + 'O'
    ser.write(off_data.encode())


def turnOnFan():
    off_data = str(0) + 'F'
    ser.write(off_data.encode())
    return

def map_space_to_degree(space):
    if space == 'desk':
        return -90
    elif space == 'bed':
        return -320
    elif space == 'bathroom':
        return 200
    elif space == 'empty':
        return 0
    return 0  # Default to 0 if space is not recognized

def updateLabel(direction):
    direction_label["text"] = direction
    if direction == "bed":
        img["image"] = bed_image
    elif direction == "desk":
        img["image"] = desk_image
    elif direction == "bathroom":
        img["image"] = bathroom_image
    else:
        img["image"] = default_image


def update_interface():
    length, y_pred = train_test()
    L = ['desk', 'bed', 'bathroom']
    i = 0
    global current_degree

    while i < length:
        if random.random() < 0.1 and i > 1:
            predicted_label = 'empty'
            off_data = str(0) + 'O'
            ser.write(off_data.encode())
        else:
            predicted_label = L[np.argmax(y_pred[i])]
            off_data = str(0) + 'F'
            ser.write(off_data.encode())
        print("Predicted Label:", i, predicted_label)
        # Update the GUI based on the predicted label
        updateLabel(predicted_label)

        degree = map_space_to_degree(predicted_label)
        if current_degree != degree:
            # Send command to Arduino
            turn_degree = degree - current_degree
            data = str(abs(turn_degree)) + ('C' if turn_degree >= 0 else 'A')
            ser.write(data.encode())
            print("Sent Data:", data)
            # Update the current degree
            current_degree = degree
        i += 1
        time.sleep(5)
        root.update()

        # Schedule the next update of the interface
    root.after(3000, update_interface)





button_frame = Frame(root)
button_frame.pack(side='left')
turn_on = Button(button_frame, text="ON", command=turnOnFan)
turn_on.pack()

turn_off = Button(button_frame, text="OFF", command=turnOffFan)
turn_off.pack()

manualControl = Button(button_frame, text="MANUAL", command=manual)
manualControl.pack()

bed = Button(button_frame, text="BED", command=pointToBed)
bed.pack()
desk = Button(button_frame, text="DESK", command=pointToDesk)
desk.pack()
bathroom = Button(button_frame, text="BATHROOM", command=pointToBathroom)
bathroom.pack()

autoControl = Button(button_frame, text="AUTO", command=auto)
autoControl.pack()
auto()
label_frame = Frame(root)
label_frame.pack(side='bottom')
direction_label = Label(label_frame, text="default", fg='black', relief=RAISED)
direction_label.pack(ipady=5, fill='x')
direction_label.config(font=("Font", 30))
img = Label(root, image=default_image)
img.pack(side='right')
update_interface()
root.mainloop()





