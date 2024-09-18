import asyncio
from queue import Queue
import websockets
import json


async def queue_listener(shared_subtitle_queue, active_connections):
    print("Queue listener started")
    try:
        while True:
            try:
                message = await asyncio.wait_for(shared_subtitle_queue.get(), timeout=0.1)
                print('queue_listener_message', message)
                # 向所有活跃连接发送消息
                for websocket in active_connections:
                    try:
                        await websocket.send(json.dumps({"subtitle": message}))
                    except websockets.exceptions.ConnectionClosed:
                        active_connections.remove(websocket)
            except asyncio.TimeoutError:
                # 超时后继续循环
                await asyncio.sleep(0)
    except asyncio.CancelledError:
        print("Queue listener task cancelled")


async def audio_server(websocket, path, q: Queue, shared_ws_message: Queue, active_connections):
    active_connections.add(websocket)
    try:
        async for message in websocket:
            shared_ws_message.put({
                "data": message,
                "type": "webm_data"
            })
    finally:
        active_connections.remove(websocket)


async def start_websocket_server(q: Queue, shared_ws_message: Queue, shared_subtitle_queue: asyncio.Queue, port=8765):
    active_connections = set()

    # 启动队列监听任务
    listener_task = asyncio.create_task(queue_listener(
        shared_subtitle_queue, active_connections))

    server = await websockets.serve(
        lambda ws, path: audio_server(
            ws, path, q, shared_ws_message, active_connections),
        "localhost",
        port
    )
    q.put('WebSocket server started')
    return server, listener_task


async def run_websocket_server(q: Queue, shared_ws_message: Queue, shared_subtitle_queue: asyncio.Queue, port=8765):
    server, listener_task = await start_websocket_server(q, shared_ws_message, shared_subtitle_queue, port)
    try:
        await server.wait_closed()
    finally:
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass
