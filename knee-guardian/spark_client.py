# spark_client.py

import websocket
import ssl
import json
import base64
import hashlib
import hmac
import threading
from urllib.parse import urlparse, urlencode
from datetime import datetime
from time import mktime
from wsgiref.handlers import format_date_time


class SparkChatClient:
    def __init__(self, app_id, api_key, api_secret, spark_url, domain):
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.spark_url = spark_url
        self.host = urlparse(spark_url).netloc
        self.path = urlparse(spark_url).path
        self.domain = domain

        # 用于线程同步和存储结果
        self.response_text = ""
        self.is_finished = threading.Event()

    def create_url(self):
        # 1. 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 2. 拼接字符串
        signature_origin = f"host: {self.host}\n"
        signature_origin += f"date: {date}\n"
        signature_origin += f"GET {self.path} HTTP/1.1"

        # 3. HMAC-SHA256加密
        signature_sha = hmac.new(self.api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        # 4. 构造授权字符串
        authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 5. 构造最终URL
        v = {"authorization": authorization, "date": date, "host": self.host}
        url = self.spark_url + '?' + urlencode(v)
        return url

    def on_message(self, ws, message):
        data = json.loads(message)
        code = data['header']['code']
        if code != 0:
            print(f'请求错误: {code}, {data}')
            self.response_text = f"AI服务返回错误，错误码: {code}"
            self.is_finished.set()  # 发生错误，结束等待
            ws.close()
            return

        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]

        # 拼接每一次返回的内容
        self.response_text += content

        # status为2代表最后一条消息
        if status == 2:
            ws.close()
            self.is_finished.set()  # 正常结束，结束等待

    def on_error(self, ws, error):
        print("### WebSocket Error:", error)
        self.response_text = "连接AI服务时发生WebSocket错误。"
        self.is_finished.set()  # 发生错误，结束等待

    def on_close(self, ws, close_status_code, close_msg):
        # print("### WebSocket Closed ###")
        if not self.is_finished.is_set():
            self.is_finished.set()  # 确保任何关闭都会触发事件

    def on_open(self, ws):
        data = json.dumps(self.gen_params(self.app_id, self.current_messages, self.domain))
        ws.send(data)

    def gen_params(self, appid, messages, domain):
        """构造请求参数，messages 为 [{"role": "system"/"user"/"assistant", "content": "..."}] 列表"""
        data = {
            "header": {"app_id": appid, "uid": "flask_user_123"},
            "parameter": {"chat": {"domain": domain, "temperature": 0.5, "max_tokens": 4096}},
            "payload": {"message": {"text": messages}}
        }
        return data

    def _send_request(self, messages):
        """发送请求并等待返回（内部方法）"""
        self.response_text = ""
        self.is_finished.clear()
        self.current_messages = messages

        ws_url = self.create_url()
        ws = websocket.WebSocketApp(ws_url,
                                    on_message=self.on_message,
                                    on_error=self.on_error,
                                    on_close=self.on_close,
                                    on_open=self.on_open)

        wst = threading.Thread(target=ws.run_forever, kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}})
        wst.daemon = True
        wst.start()

        finished_in_time = self.is_finished.wait(timeout=60)
        if not finished_in_time:
            ws.close()
            return "请求AI服务超时。"

        return self.response_text

    def get_response(self, query):
        """单轮对话（向后兼容）"""
        return self._send_request([{"role": "user", "content": query}])

    def chat(self, messages):
        """多轮对话，messages 列表包含完整上下文，最后一条为用户最新问题"""
        return self._send_request(messages)
