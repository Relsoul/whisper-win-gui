from queue import Queue
import threading
import numpy as np
import soundfile as sf
import pyaudiowpatch as pyaudio
import time
from pydub import AudioSegment
from scipy.signal import resample
from scipy import signal
import pdb  # 导入调试模块
import torch
import os
from transformers import pipeline
from copy import deepcopy
