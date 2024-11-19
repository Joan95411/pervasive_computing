# Smart Home Heating System

This is a Python program for WARMFLO. The system integrates with an Arduino board to control the heating flow based on predicted occupancy detection, as well as a graphical user interface (GUI) for manual control.

### Features

* **Manual and Auto Mode**: The system supports both manual and automatic modes. In manual mode, users can manually control the heating system for WARMFLO's heating flow direction and fan on/off switch, while in auto mode, the system automatically adjusts the heating based on occupancy detection.

* **Occupancy Detection**: The system uses machine learning algorithms to detect room occupancy using channel state data. It predicts the occupancy status of rooms (e.g., bedroom, desk, bathroom) and adjusts the heating accordingly.

* **Graphical User Interface**: The system includes a GUI built using Tkinter, allowing users to interact with the heating system easily. Users can switch between manual and auto mode, control individual room heating, and view the current status of the system.

### Requirements
* Python 3.x
* Tkinter (included in standard Python distributions)
* NumPy
* Serial (for communication with Arduino)
* TensorFlow 
* Keras 
* Scikit-learn

### Usage
1. Connect the Arduino board to the computer and configure the serial port settings in the code (COM3, 9600 baud rate).
2. Run the main.py file:
`python main.py`
3. The test.py starts training the data from 'data' file, then send the test data for occupant presence to main.py as it sends command to arduino for direction turning, and then update frames in user interface.
4. The GUI window will open. Use the buttons to control the heating system:
   * Click "MANUAL" to switch to manual mode and control individual room heating.
   * Click "AUTO" to switch to automatic mode and let the system adjust heating based on occupancy detection.
   * Use the "BED", "DESK", and "BATHROOM" buttons to direct the heating towards specific rooms.
   * Use the "ON" and "OFF" buttons to manually control the heating fan.
   Follow the on-screen instructions to interact with the system.

### Acknowledgement
Many thanks to Jeroen Klein Brinke for his invaluable assistance with the WiFi human activity detection aspect of our project. His expertise and guidance have been instrumental in shaping our research and contributing to its success. We are grateful for his support and collaboration throughout this endeavor.