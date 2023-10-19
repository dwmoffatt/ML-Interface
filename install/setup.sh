#!/bin/bash

echo "--- Installing Dependencies ---"

export QT_QPA_PLATFORM="xcb"

sudo apt update
sudo apt -y upgrade

sudo apt install -y libxcb-xinerama0 libqt5x11extras5

pip3 install --upgrade pip
pip3 install -r ../requirements.txt
