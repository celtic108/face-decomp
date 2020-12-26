#!/usr/bin/env bash
FROM tensorflow/tensorflow:2.4.0-gpu

COPY . .

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install Pillow
RUN python3 -m pip install -r requirements.txt
