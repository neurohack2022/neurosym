# NeuroSymphony Windows Installation

Step 1:
Connect the Muse 2 to your Windows device using the dongle provided with the Muse 2. Confirm that the Muse 2 is connected to your Windows device by looking it up on the Device Manager through the Control Panel; it should show up on one of the COM ports.

Step 2: 
Clone the files on the Neurosymphony Github repository to your device.
```sh
git clone https://github.com/neurohack2022/neurosym
```
Step 3:
Create a virtual environment to install the dependencies:
```sh
python -m venv neurosym
```
Step 4:
Activate the virtual environment and install the dependencies for Windows
```sh
./neurosym/Scripts/Activate.ps1
pip install -r requirements.txt
```
Step 5:
Run main_menu.py to start the program
```sh
python ./main_menu.py
```
Step 6:
After the main menu pops up, select the data type as 'Task Live' and click on the Graph button. A Graph Window with the EEG data pops up. Shortly after, a Google Chrome window with sliders adjusting the volume of different soundscapes pops up and so does a window with 'Train' and 'Run' buttons. Click on the 'Train' button, this trains the model for a certain amount of episodes which may take a while. After the training is completed click on the 'Run' button

Step 7:
To end the program close all the windows
