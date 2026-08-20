# UR5 Pick and Place

<b>Author:</b> <a href="https://orcid.org/my-orcid?orcid=0009-0006-2253-4195" target="_blank">Allan Almeida</a>

## Overview

This project implements a pick and place application with a UR5e robot arm on Webots.

Forward and inverse kinematics and all control functions are implemented in Python. The robot is controlled
by sending joint velocities to the robot using a quintic polynomial trajectory.
The robot is able to pick up a bottle from a table
and place it on another table (and also give the dummy a drink :grin: :beer:).

The computer vision part of the project is implemented via a CNN. The CNN is trained
to detect the bottle position on the image. The CNN is implemented in Tensorflow and Keras,
and it uses a VGG16 pre-trained model as a base. The CNN is modified to predict the bottle position relative to the image and convert it to the real XYZ coordinates using bilinear interpolation. It is trained and evaluated on a dataset of 5000 images.

<a href="https://youtu.be/XgvYlSNmqiI">Demo video</a>

## Dependencies

- Webots
- Python >= 3.6
- Jupyter Notebook (Anaconda or pip)

## Usage

1. Open Webots and load the world `my_first_simulation.wbt`. Alternatively, you can launch Webots and load the world from the command line simply by running:
   ```
   ./launch.sh
   ```
2. Create a virtual environment and install the dependencies from `requirements.txt`
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook `Trabalho.ipynb` and run the first cell to import the dependencies and start the simulation
4. Run the other cells, one by one, to see the robot in action
5. You can use the functions of `ur5.py` to control the robot and perform other tasks you want

Have fun! :sparkles: :robot:
