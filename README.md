#Recipe generator service
This project's goal is to combine YOLO and BitNet to provide a recipe creating service, YOLO will be used detect ingredients from an image and the BitNet will generate a recipe using these ingredients.

## Installation
1. Clone the repository:
git clone https://github.com/5CCSACCA/coursework-n-sroat

## Download model weights
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T

## Running
python main.py
