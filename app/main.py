from ultralytics import YOLO
from run_inference import run_inference
import argparse
import subprocess
from contextlib import redirect_stdout
import os
import pexpect
import sys
from ctypes import cdll, c_char_p

lib = cdll.LoadLibrary("./build/bin/libbitnet.so")

# Return type for string-returning functions
lib.bitnet_get_last_output.restype = c_char_p

# Argument type for functions that take a C string
lib.bitnet_set_python_prompt.argtypes = [c_char_p]

# Set a prompt
lib.bitnet_set_python_prompt(b"Hello Python!")

# Get last output
output = lib.bitnet_get_last_output()
print(output.decode())  # decode from bytes to string

# Clear output and prompt
lib.bitnet_clear_last_output()
lib.bitnet_clear_python_prompt()

'''
model = YOLO("yolo11n.pt")

results = model("https://ultralytics.com/images/bus.jpg")

for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box

'''
cmd = ["python", "-u" ,"run_inference.py"]
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
child = pexpect.spawn(
    "python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p 'You are a helpful assistant' -cnv",
    encoding="utf-8"
)

child.timeout = None  # wait forever if needed
captured_output = ""
while True:
    try:
        chunk = child.read_nonblocking(size=1024, timeout=1)
        print(chunk, end="")      # live output
        captured_output += chunk  # store
    except pexpect.EOF:
        break
    except pexpect.TIMEOUT:
        continue  # no data yet, loop again