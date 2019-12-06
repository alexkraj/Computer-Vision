COMPUTER VISION - ASSIGNMENT 2019/2020 CTVT58
===================================================================================

_____________________________1. Running the Prototype_____________________________

***To run the program on Linux/Mac/Mira:***
*note that Mira can be accessed on the university computers using the windows
command prompt by using abcd12@mira.dur.ac.uk and typing in your password*

Ensure that python3 is installed in the path.
Ensure that all the requisite libraries are installed in the path:
- numpy
- opencv

Ensure that the directory structure is correct. The program main.py accesses 2 folders
of data, namely:

../data/
../masks/

- ../data/ is a directory containing left and right PNGs from a stereo recording
- ../masks/ is a directory containing left and right binary masks (left.png and 
right.png respectfully)

Both of these variables can be changed in the main.py program under the following 
variable names:
master_path_to_dataset
left_mask
right_mask
