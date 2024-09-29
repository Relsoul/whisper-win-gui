import asyncio
from copy import deepcopy
import os
import time
import flet as ft
import threading
import queue
from lib.model import start_model_server
# from lib.watch_win_system_audio import start_watch_win_system_audio
from lib.ws import start_websocket_server
from datetime import datetime
import soundfile as sf
import numpy as np
import atexit
from queue import Queue
import pyaudiowpatch as pyaudio


class SharedState:
    def __init__(self):
        self.ws_print = Queue()
        self.print_queue = Queue()
        self.subtitle_queue = Queue()
        self.subtitle_print_queue = Queue()
        self.audio_queue = Queue()
        self.ws_subtitle_queue = asyncio.Queue()
        self.condition = threading.Condition()
        self.stop_event = threading.Event()


shared_state = SharedState()


class GlobalButtonLock:
    def __init__(self):
        self.ws_button_lock = False
        self.model_button_lock = False
        self.audio_button_lock = False


global_button_lock = GlobalButtonLock()


def format_message(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    s = f"[{current_time}] {message} \r\n"
    print(s)
    return s


def save_message2file(message):
    with open('message.txt', 'a', encoding='utf-8') as f:
        f.write(message)


def process_queues(outputArea: ft.Text, subtitleOutPutArea: ft.Text):
    with shared_state.condition:
        if not shared_state.ws_print.empty():
            message = shared_state.ws_print.get()
            outputArea.value += format_message(message)
            return True
        if not shared_state.print_queue.empty():
            message = shared_state.print_queue.get()
            outputArea.value += format_message(message)
            return True
        if not shared_state.subtitle_print_queue.empty():
            message = shared_state.subtitle_print_queue.get()
            fmsg = format_message(message)
            save_message2file(fmsg)
            subtitleOutPutArea.value = fmsg + subtitleOutPutArea.value
            return True
    return False


async def process_subtitle_queue(outputArea: ft.Text):
    with shared_state.condition:
        if not shared_state.subtitle_queue.empty():
            message = shared_state.subtitle_queue.get()
            shared_state.subtitle_print_queue.put(message)
            if global_button_lock.ws_button_lock:
                await shared_state.ws_subtitle_queue.put(message)
            outputArea.value = format_message(message)
            return True
    return False


def queue_monitor(page: ft.Page, outputArea: ft.Text, subtitleOutPutArea: ft.Text):
    while not shared_state.stop_event.is_set():
        with shared_state.condition:
            while not process_queues(outputArea, subtitleOutPutArea):
                if shared_state.condition.wait(timeout=0.1):
                    break
        page.update()


async def subtitle_queue_monitor(page: ft.Page, outputArea: ft.Text):
    while not shared_state.stop_event.is_set():
        updated = await process_subtitle_queue(outputArea)
        if updated:
            await page.update_async()
        else:
            await asyncio.sleep(0.1)  # 短暂休眠以避免过度消耗 CPU


async def start_websocket(e):
    if global_button_lock.ws_button_lock:
        return
    global_button_lock.ws_button_lock = True

    try:
        server, listener_task = await start_websocket_server(
            q=shared_state.ws_print,
            shared_ws_message=shared_state.audio_queue,
            shared_subtitle_queue=shared_state.ws_subtitle_queue
        )
        shared_state.print_queue.put('WebSocket server started')

        # 等待服务器关闭
        await server.wait_closed()
    except Exception as e:
        shared_state.print_queue.put(f'WebSocket server error: {str(e)}')
    finally:
        # 确保监听器任务被取消
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass
        global_button_lock.ws_button_lock = False


async def start_model(e):
    if global_button_lock.model_button_lock:
        return
    global_button_lock.model_button_lock = True

    thread = threading.Thread(target=start_model_server, kwargs={
        "print_queue": shared_state.print_queue,
        "shared_audio_queue": shared_state.audio_queue,
        "shared_subtitle_queue": shared_state.subtitle_queue,
    })
    thread.start()


async def start_window_system_audio(e):
    # audio 原生不支持线程 需要在主线程初始化audio.open
    # thread = threading.Thread(target=start_watch_win_system_audio, kwargs={
    #     "print_queue": shared_state.print_queue,
    #     "shared_audio_queue": shared_state.audio_queue,
    # })
    # thread.start()
    # Get default WASAPI info
    # 启动进程 而不是线程
    # wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    # print('wasapi_info', wasapi_info)
    # start_watch_win_system_audio(p=p, wasapi_info=wasapi_info, print_queue=shared_state.print_queue,
    #                              shared_audio_queue=shared_state.audio_queue)

    if global_button_lock.audio_button_lock:
        return
    global_button_lock.audio_button_lock = True

    await start_watch_win_system_audio()
    pass


async def test_audio(e):
    audio_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), './zh2.wav')
    audio_file = os.path.normpath(audio_file)
    # audio = whisper.load_audio()
    audio_data, sample_rate = sf.read(audio_file, dtype=np.float32)
    shared_state.audio_queue.put({
        "data": audio_data,
        "samplerate": sample_rate,
        "type": "nparray"
    })
    print('test_audio push data 2 queue done', sample_rate)


async def test_put_ws(e):
    await shared_state.ws_subtitle_queue.put('test_put_ws')
    print('test_put_ws push data 2 queue done')


async def exit_app(page):
    print('exit_app')
    page.window.close()
    exit()


async def main(page: ft.Page):
    page.add(ft.Text("点击下述按钮启动对应服务，注意：model服务需要先启动"))
    row = ft.Row([
        ft.ElevatedButton(text="启动model服务", on_click=start_model),
        ft.ElevatedButton(text="启动websocket服务", on_click=start_websocket),
        ft.ElevatedButton(text="启动监听windows系统音频",
                          on_click=start_window_system_audio),
        ft.ElevatedButton(text="test audio", on_click=test_audio),
        ft.ElevatedButton(text="test put ws", on_click=test_put_ws)
    ])
    page.add(row)
    # page.add(ft.ElevatedButton(text="启动websocket服务", on_click=start_websocket))
    # page.add(ft.ElevatedButton(text="启动model服务", on_click=start_model))
    # page.add(ft.ElevatedButton(text="启动监听windows系统音频",
    #          on_click=start_window_system_audio))
    # page.add(ft.ElevatedButton(text="test audio", on_click=test_audio))

    page.add(ft.Divider())

    outputArea = ft.Text(value="", height=300, width=500)

    subtitleOutPutArea = ft.Text(value="", height=300, width=500)

    outRow = ft.Row([
        ft.Column([
            ft.Container(ft.Text("系统输出区", color=ft.colors.WHITE,),
                         bgcolor=ft.colors.AMBER_300,  padding=10),
            outputArea
        ]),
        ft.Column([
            ft.Container(ft.Text("字幕输出区"),
                         bgcolor=ft.colors.AMBER_300, padding=10),
            subtitleOutPutArea,
        ], scroll=ft.ScrollMode.ALWAYS)
    ])

    page.add(outRow)

    # def on_window_event(e):
    #     print('on_window_event', e)

    # page.window.on_event = on_window_event

    # async def on_close(e):
    #     await exit_app(e)
    #     shared_state.stop_event.set()
    #     with shared_state.condition:
    #         shared_state.condition.notify_all()
    #     monitor_thread.join()

    # page.on_close = on_close

    # 启动队列监视器线程
    monitor_thread = threading.Thread(
        target=queue_monitor, args=(page, outputArea, subtitleOutPutArea))
    monitor_thread.start()

    # 确保在应用关闭时停止监视器线程


DURATION = 999999999
# CHUNK_SIZE = 512
CHUNK_SIZE = 2048  # 采样点
Channels = 6
sample_rate = 48000
_array = []
time_gap = 1.3


def cb(in_data, frame_count, time_info, status):
    # print('监听数据', in_data)
    if status:
        print(status)
    # print(in_data)
    global _array, Channels, sample_rate
    # print('cb', Channels, sample_rate)
    # 将输入音频转换为6声道浮点数
    audio_array = np.frombuffer(
        in_data, dtype=np.float32).reshape(-1, Channels)
    # 确保输入数据格式和采样率一致后再进行处理
    # audio_array = transform_audio_data(audio_array, sample_rate)
    global _array
    # 判断是否有音频数据
    if len(audio_array) > 0:
        _array.append(audio_array.copy())

    # _array.append(in_data)
    return (in_data, pyaudio.paContinue)


_prev_array = []


def timer_func():
    print('分割系统音频 put队列')
    global _array, sample_rate, _prev_array
    # audio_data = np.concatenate(_array)
    # audio_data = b''.join(_array)
    array_copy = deepcopy(_array)

    _array = []
    if len(array_copy) == 0:
        print('no audio data')
        # 重新启动定时器
        timer = threading.Timer(time_gap, timer_func)
        timer.start()
        return
    _prevlen = len(_prev_array)
    if _prevlen > 0:
        # 拼接20% 可以保证音频的连续性
        p_concat = np.concatenate(_prev_array[-int(_prevlen*0.2):])
        # print('p_concat', p_concat)
        array_copy_contact = np.concatenate(array_copy)
        audio_data = np.vstack((p_concat, array_copy_contact))
        # print('combined_array', combined_array)
        # audio_data = np.concatenate(
        #     [_prev_array[-int(_prevlen*0.2):], array_copy])
    else:
        audio_data = np.concatenate(array_copy)

    _prev_array = array_copy
    shared_state.audio_queue.put({
        "data": audio_data,
        "samplerate": sample_rate,
        "type": "nparray"
    })
    # 清空已经处理的_array音频数据 而不是直接clean整个_array
    # _array.pop(0)

    # 重新启动定时器
    timer = threading.Timer(time_gap, timer_func)
    timer.start()


def start_timer():
    timer = threading.Timer(time_gap, timer_func)
    timer.start()


async def start_watch_win_system_audio():
    p = pyaudio.PyAudio()

    # with pyaudio.PyAudio() as p:
    #     """
    #     Create PyAudio instance via context manager.
    #     Spinner is a helper class, for `pretty` output
    #     """
    try:
        # Get default WASAPI info
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError as e:
        print(e)
        shared_state.print_queue.put(
            "Looks like WASAPI is not available on the system. Exiting...")
        print(
            "Looks like WASAPI is not available on the system. Exiting...")
        exit()
    # ge = p.get_loopback_device_info_generator()
    # Open PyA manager via context manager
    default_speakers = p.get_device_info_by_index(
        wasapi_info["defaultOutputDevice"])
    if not default_speakers["isLoopbackDevice"]:
        for loopback in p.get_loopback_device_info_generator():
            """
            Try to find loopback device with same name(and [Loopback suffix]).
            Unfortunately, this is the most adequate way at the moment.
            """
            if default_speakers["name"] in loopback["name"]:
                default_speakers = loopback
                break
        else:
            print("Default loopback output device not found.\n\nRun `python -m pyaudiowpatch` to check available devices.\nExiting...\n")
            shared_state.print_queue.put(
                "Default loopback output device not found.\n\nRun `python -m pyaudiowpatch` to check available devices.\nExiting...\n")
            exit()
    print(f"Default speakers: {default_speakers['name']}")
    shared_state.print_queue.put(
        f"Default speakers: {default_speakers['name']}")
    global Channels, sample_rate
    Channels = default_speakers["maxInputChannels"]
    sample_rate = int(default_speakers["defaultSampleRate"])
    print('sample_rate', sample_rate)
    print('Channels', Channels)
    shared_state.print_queue.put(f"Channels: {Channels}")
    shared_state.print_queue.put(f"sample_rate: {sample_rate}")
    with p.open(format=pyaudio.paFloat32,
                channels=Channels,
                # rate=int(default_speakers["defaultSampleRate"]),
                rate=sample_rate,
                frames_per_buffer=CHUNK_SIZE,
                input=True,
                input_device_index=default_speakers["index"],
                stream_callback=cb
                ) as stream:
        """
        Opena PA stream via context manager.
        After leaving the context, everything will
        be correctly closed(Stream, PyAudio manager)
        """
        print(f"The next {DURATION} seconds will be written to ")
        shared_state.print_queue.put(
            f"The next {DURATION} seconds will be written to ")
        start_timer()
        await asyncio.sleep(DURATION)  # Blocking execution while playing
    pass


async def sub_title_window(page: ft.Page):
    page.window.title_bar_hidden = True
    page.window.title_bar_buttons_hidden = True
    page.window.always_on_top = True
    page.window.frameless = True
    page.opacity = 0.5
    # 设置半透明黑色背景
    background_color = ft.colors.TRANSPARENT
    page.window.bgcolor = background_color
    page.bgcolor = ft.colors.with_opacity(0.5, ft.colors.BLACK)

    page.window.width = 400
    page.window.height = 120

    page.add(
        ft.Row(
            [
                ft.WindowDragArea(ft.Container(ft.Text(
                    "点击此区域拖动", color=ft.colors.WHITE), bgcolor=ft.colors.AMBER_300, padding=10, ), expand=True),
                ft.IconButton(ft.icons.CLOSE,
                              icon_color=ft.colors.WHITE,
                              on_click=lambda _: page.window.close())
            ]
        )
    )
    outputArea = ft.Text(value="", height=130, width=500,
                         color=ft.colors.WHITE)
    page.add(outputArea)
    # 启动队列监视器线程
    asyncio.create_task(subtitle_queue_monitor(page, outputArea))

    # monitor_thread = threading.Thread(
    #     target=subtitle_queue_monitor, args=(page, outputArea))
    # monitor_thread.start()


async def start_sub_title_window():
    await ft.app_async(sub_title_window)


async def start_main():
    await ft.app_async(main)


# 主协调函数
async def run_all_tasks():
    # 创建任务列表
    tasks = [
        start_main(),
        start_sub_title_window(),
        # 可以在这里添加更多的异步任务
    ]

    # 同时运行所有任务
    await asyncio.gather(*tasks)

if __name__ == "__main__":

    # start_watch_win_system_audio(p=p)
    # 启动协程
    # asyncio.gather(start_main())
    asyncio.run(run_all_tasks())
    print('exit all window')
    os._exit(0)
