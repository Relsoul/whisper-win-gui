from queue import Queue
from typing import Literal
import numpy as np
import io
import soundfile as sf
import torch
import os
from pydub import AudioSegment
from scipy.signal import resample
from scipy import signal
from transformers import pipeline
import pyaudiowpatch as pyaudio
from datasets import Dataset
import time


# Function to convert WebM to WAV
transcriber = None


def convert_webm_to_wav(webm_data):
    # Load WebM audio from a byte stream (e.g., from WebSocket data)
    audio = AudioSegment.from_file(io.BytesIO(webm_data), format="webm")
    # save to file
    # audio.export('zh2.wav', format='wav')

    # Export as WAV to a byte buffer
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    return wav_io

# Function to read WAV data into NumPy array using soundfile


def read_wav_as_np(wav_io):
    # Read the WAV file from the byte buffer
    data, samplerate = sf.read(wav_io, dtype=np.float32)
    return data, samplerate


# def transform_audio_data(audio_data, sample_rate):
#     if len(audio_data.shape) > 1:
#         # 通过平均各声道的数据，将音频转换为单声道
#         audio_data = np.mean(audio_data, axis=1)
#         # 将音频采样率调整为16kHz，Whisper期望的采样率
#     target_sample_rate = 16000
#     if sample_rate != target_sample_rate:
#         num_samples = int(len(audio_data) * target_sample_rate / sample_rate)
#         audio_data = resample(audio_data, num_samples)
#     # 添加高通滤波器去除低频噪音
#     sos = signal.butter(10, 100, 'hp', fs=target_sample_rate, output='sos')
#     audio_data = signal.sosfilt(sos, audio_data)

#     # 音量归一化
#     # audio_data = audio_data / np.max(np.abs(audio_data))
#     return audio_data

def transform_audio_data(audio_data, sample_rate):
    target_sample_rate = 16000

    # Step 1: 使用更宽频段的高通滤波，避免失真
    sos = signal.butter(4, 80, 'hp', fs=sample_rate,
                        output='sos')  # 阶数降低为4，频率降低为80Hz
    filtered_audio_data = signal.sosfilt(sos, audio_data, axis=0)

    # Step 2: 对每个声道单独进行高质量重采样
    resampled_audio_data = []
    for channel in range(filtered_audio_data.shape[1]):
        resampled_channel = signal.resample_poly(
            filtered_audio_data[:, channel], target_sample_rate, sample_rate)
        resampled_audio_data.append(resampled_channel)

    # 将各个声道重新组合成多声道数据
    resampled_audio_data = np.array(
        resampled_audio_data).T  # 转置成 [samples, channels]

    # Step 3: 将重采样后的音频转换为单声道（对所有声道取平均）
    mono_audio_data = np.mean(resampled_audio_data, axis=1)

    # step4: 音量归一化 可能需要判断多声道的情况下是否需要归一化
    mono_audio_data = mono_audio_data / np.max(np.abs(mono_audio_data))
    return mono_audio_data


def process_audio(data, samplerate, print_queue: Queue, type: Literal['webm_data', 'nparray'] = 'webm_data',):
    # wirte audio data to file
    # audio_data 为float32 list数据 [-0.0042600203305482864, -0.0044130501337349415, -0.0034505745861679316, -0.003668916644528508, -0.003148190677165985, -0.00296563352458179, -0.002432041335850954, -0.0022274774964898825, -0.002190340543165803, -0.0015896748518571258, -0.0011883934494107962, -0.0010361482854932547, -0.0009721876122057438, -0.0005577248521149158, -0.0005685987416654825, -0.0010715476237237453, -0.001034380286000669, -0.0008596990373916924, -0.0010461580241099, -0.0011197144631296396, -0.0010246435413137078, -0.0013365905033424497]
    # print('audio_data', audio_data)
    # audio_data = np.array(audio_data, dtype=np.float32)
    audio_data = data
    if type == 'webm_data':
        try:
            wav_io = convert_webm_to_wav(data)
            x_audio_data, x_samplerate = read_wav_as_np(wav_io)
            audio_data = x_audio_data
            samplerate = x_samplerate
        except Exception as e:
            print('convert_webm_to_wav error', e)
            return ''
    print('wav_io done')

    audio_data = transform_audio_data(audio_data, samplerate)
    # sf.write('demo.wav', audio_data, 16000, format='wav')

    # print('transform_audio_data audio_data done')
    print_queue.put('transform_audio_data audio_data done')

    # audio_data = np.array(audio_data, dtype=np.float32)
    # audio_data = (audio_data).astype(np.float32)

    # sf.write('zh1.wav', audio_data, 48000, format='wav')

    # 将float数据转换为32位整数

    # audio_data = (audio_data).astype(np.float32)

    # 将音频数据写入内存缓冲区
    # buffer = io.BytesIO()
    # sf.write(buffer, audio_data, 44100, format='wav')
    # buffer.seek(0)

    # 使用whisper进行语音识别
    # Read audio data from buffer directly, use np.frombuffer to convert it into array
    # audio_array, _ = sf.read(buffer)
    # # 将内存缓冲区的内容写入临时文件
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
    #     temp_file.write(buffer.getvalue())
    #     temp_file_path = temp_file.name
    # print("temp_file_path", temp_file_path)
    # audio = whisper.load_audio('./zh1.wav')
    global transcriber
    # 创建一个Dataset对象
    # dataset = Dataset.from_dict({"audio": [audio_data]})

    # 使用map函数进行批处理
    # results = dataset.map(lambda x: transcriber(
    #     x["audio"]), batched=True, batch_size=8)
    result = transcriber(audio_data)
    # print('results', results)
    # 提取字幕文本
    subtitle = result["text"]
    print('subtitle', subtitle)
    print_queue.put(f'subtitle: ${subtitle}')

    # pass
    return subtitle


def load_model_by_pipe(device: Literal['cpu', 'cuda', 'cuda:0'], print_queue: Queue):
    # 初始化Whisper模型
    # model_size = "medium"  # 选择合适的模型
    print("Loading model...")
    print_queue.put('Loading model...')
    local_model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '../wmodel/')
    local_model_path = os.path.normpath(local_model_path)

    print('local_model_path', local_model_path, device)
    global transcriber
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=local_model_path,
        device=device,
        generate_kwargs={
            # "num_beams": 3,  # 使用 beam search
            # "temperature": 0,  # 调整温度
            "condition_on_prev_tokens": False,
        },
        # batch_size=8,  # 调整批处理大小
        # chunk_length_s=10
    )
    print("Model loaded")
    transcriber.model.config.forced_decoder_ids = (
        transcriber.tokenizer.get_decoder_prompt_ids(
            language="zh",
            task="transcribe"
        )
    )

    print_queue.put('Model loaded')


# 待完善，目前使用transformers的pipeline方式
# def load_model_by_whisper(device: Literal['cpu', 'cuda', 'cuda:0'], print_queue: Queue):
    # model_size = "medium"  # 选择合适的模型
    # print("Loading model...")
    # print_queue.put('Loading model...')
    # # current file path+../whisper_model/Belle-whisper-large-v3-zh-punct
    # local_model_path = os.path.join(
    #     os.path.dirname(os.path.abspath(__file__)), '../whisper_model')
    # # normalize path
    # local_model_path = os.path.normpath(local_model_path)
    # print('local_model_path', local_model_path)

    # # model_size = os.path.basename(local_model_path)  # 假设文件夹名就是模型大小
    # model = whisper.load_model(name='large',
    #                            download_root=local_model_path, device=device, in_memory=True)
    # print("Model loaded")
    # print_queue.put('Model loaded')
    # return model


def start_model_server(print_queue: Queue, shared_audio_queue: Queue, shared_subtitle_queue: Queue,):
    cuda = torch.cuda.is_available()
    print(f'torch cuda: ${cuda}')
    print_queue.put(f'torch cuda: ${cuda}')
    # must be transform to whisper model so this version is not use
    # load_model_by_whisper(
    #     device=cuda and 'cuda:0' or 'cpu', print_queue=print_queue)

    load_model_by_pipe(device=cuda and 'cuda:0' or 'cpu',
                       print_queue=print_queue)

    while True:
        if not shared_audio_queue.empty():
            audioDict = shared_audio_queue.get()
            subtitle = ''
            try:
                subtitle = process_audio(
                    data=audioDict['data'], samplerate=audioDict.get('samplerate') or None, type=audioDict['type'] or 'webm_data', print_queue=print_queue)
            except Exception as e:
                print('process_audio error', e)
                print_queue.put(f'process_audio error: ${e}')
            if subtitle:
                shared_subtitle_queue.put(subtitle)
    pass
