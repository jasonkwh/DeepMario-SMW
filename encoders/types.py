from enum import Enum


class Encoder(str, Enum):
    CNN = "cnn"
    RAM = "ram"
    CNN_YOLO = "cnn-yolo"
