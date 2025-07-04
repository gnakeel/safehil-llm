import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)

class WebSocketClient:
    def __init__(self, uri: str, reconnect_delay: int = 3):
        self._uri = uri
        self._websocket = None
        self._send_queue = asyncio.Queue()
        self._sender_task = None
        self._reconnect_delay = reconnect_delay
        self._connected_event = asyncio.Event()  # 연결 상태를 외부에서 알기 위함
        self._running = True  # 종료 플래그

    async def _sender_loop(self):
        """
        큐에서 메시지를 꺼내 WebSocket 으로 전송하는 메인 루프.
        연결이 끊어지면 재연결 시도.
        """
        while self._running:
            # 연결이 없으면 재연결
            if not self._websocket or self._websocket.closed:
                await self._attempt_reconnect()

            message = await self._send_queue.get()

            try:
                await self._websocket.send(json.dumps(message))
                logging.debug(f"[WebSocket] Sent: {message}")
            except websockets.exceptions.ConnectionClosed:
                logging.warning(f"[WebSocket] Connection lost, reconnecting...")
                await self._send_queue.put(message)  # 메시지 다시 넣기
                self._websocket = None
                await asyncio.sleep(self._reconnect_delay)
            except Exception as e:
                logging.error(f"[WebSocket] Send error: {e}")
                await self._send_queue.put(message)
                self._websocket = None
                await asyncio.sleep(self._reconnect_delay)

    async def _attempt_reconnect(self):
        """
        연결 재시도 루프.
        """
        while self._running:
            try:
                self._websocket = await websockets.connect(self._uri)
                self._connected_event.set()
                logging.info(f"[WebSocket] Connected to {self._uri}")
                return
            except (ConnectionRefusedError, websockets.exceptions.WebSocketException, OSError) as e:
                logging.error(f"[WebSocket] Connection failed: {e}. Retrying in {self._reconnect_delay}s...")
                self._connected_event.clear()
                await asyncio.sleep(self._reconnect_delay)
            except Exception as e:
                logging.critical(f"[WebSocket] Unexpected error: {e}")
                self._connected_event.clear()
                await asyncio.sleep(self._reconnect_delay)

    async def send_message(self, message: dict):
        """
        메시지를 전송 큐에 추가.
        """
        await self._send_queue.put(message)
        logging.debug(f"[WebSocket] Queued message: {message}")

    async def connect(self):
        """
        전송 태스크 시작.
        """
        if not self._sender_task or self._sender_task.done():
            self._running = True
            self._sender_task = asyncio.create_task(self._sender_loop())
            logging.info(f"[WebSocket] Sender loop started.")

    async def disconnect(self):
        """
        안전하게 종료.
        """
        self._running = False

        if self._sender_task:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                logging.info("[WebSocket] Sender loop cancelled.")

        if self._websocket and not self._websocket.closed:
            await self._websocket.close()
            logging.info(f"[WebSocket] Closed connection to {self._uri}")
        self._websocket = None
        self._connected_event.clear()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def wait_until_connected(self):
        """
        외부 코드가 연결 완료를 기다릴 때 사용.
        """
        await self._connected_event.wait()
