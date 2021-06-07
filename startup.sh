#!/bin/bash
apt-get install libgtk2.0-dev
apt install -y libsm6 libxext6
apt-get install -y libxrender-dev
gunicorn --bind=0.0.0.0 --timeout 600 index:server
