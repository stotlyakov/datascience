#!/bin/bash
#apt-get install -y libsm6 libxext6 libxrender-dev
#apt-get install libgtk2.0-dev
#apt-get install -y libglib2.0-0
#apt-get install -y libxrender-dev
#apt-get install ffmpeg libsm6 libxext6  -y
#apt-get install -y python3-opencv
#pip install opencv-python
pip install -U opencv-python
apt-get upgrade
apt update && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx
gunicorn --bind=0.0.0.0 --timeout 600 index:server
