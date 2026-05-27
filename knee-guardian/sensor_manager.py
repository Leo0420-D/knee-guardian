import asyncio
import threading
import time
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
import numpy as np
from array import array

# ==============================================================================
# 传感器配置 - 【【【重要】】】请根据你的设备修改这里的MAC地址
# ==============================================================================
# 在Windows上，这通常是蓝牙设置中显示的地址
# 在Linux/macOS上，可能需要用蓝牙扫描工具找到它
PAR_DEVICE_ADDR = "73:53:44:bb:9f:2e"  # <--- 在这里填入你的传感器的MAC地址

# 设备的特征UUID
PAR_NOTIFICATION_CHARACTERISTIC = "0000ae02-0000-1000-8000-00805f9b34fb"
PAR_WRITE_CHARACTERISTIC = "0000ae01-0000-1000-8000-00805f9b34fb"


# ==============================================================================
# SensorManager 类
# ==============================================================================

class SensorManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, device_address=PAR_DEVICE_ADDR):
        if not hasattr(self, 'initialized'):
            with self._lock:
                self.device_address = device_address
                self.latest_data = {}
                self.connection_status = "Disconnected"
                self.data_lock = threading.Lock()
                self.ble_thread = None
                self.initialized = True
                print("SensorManager initialized.")

    def _notification_handler(self, characteristic: BleakGATTCharacteristic, data: bytearray):
        """处理收到的蓝牙通知数据"""
        self._parse_imu(data)

    def _parse_imu(self, buf):
        """解析IMU数据并更新latest_data"""
        if buf[0] != 0x11:
            return

        ctl = (buf[2] << 8) | buf[1]
        L = 7

        # 我们只关心姿态角 (0x0040)
        if ((ctl & 0x0040) != 0):
            try:
                scaleAngle = 0.0054931640625  # 角度 [-180~+180] 180/32768
                angleX = np.short((np.short(buf[L + 1]) << 8) | buf[L]) * scaleAngle
                angleY = np.short((np.short(buf[L + 3]) << 8) | buf[L + 2]) * scaleAngle
                angleZ = np.short((np.short(buf[L + 5]) << 8) | buf[L + 4]) * scaleAngle

                with self.data_lock:
                    self.latest_data['timestamp'] = time.time()
                    self.latest_data['angleX'] = angleX  # Roll
                    self.latest_data['angleY'] = angleY  # Pitch - 我们可能用这个来表示大腿倾斜
                    self.latest_data['angleZ'] = angleZ  # Yaw
            except IndexError:
                # 数据包不完整，忽略
                pass

    async def _run_ble_loop(self):
        """异步的BLE主循环"""
        while True:
            self.connection_status = f"Scanning for {self.device_address}..."
            print(self.connection_status)
            device = await BleakScanner.find_device_by_address(self.device_address, timeout=10.0)

            if not device:
                print(f"Device {self.device_address} not found. Retrying in 5s...")
                await asyncio.sleep(5)
                continue

            self.connection_status = f"Connecting to {self.device_address}..."
            print(self.connection_status)

            async with BleakClient(device) as client:
                if client.is_connected:
                    self.connection_status = "Connected"
                    print(f"Successfully connected to {self.device_address}")

                    try:
                        # 启动通知
                        await client.start_notify(PAR_NOTIFICATION_CHARACTERISTIC, self._notification_handler)

                        # 发送配置指令 (与ble_test.py中的相同)
                        # 保持连接
                        await client.write_gatt_char(PAR_WRITE_CHARACTERISTIC, bytes([0x29]), response=True)
                        await asyncio.sleep(0.2)

                        # 设置主动上报数据
                        Cmd_ReportTag = 0x0040  # 只订阅姿态角(0x0040)，节省带宽
                        params = bytearray(
                            [0x12, 5, 255, 0, 0, 60, 1, 3, 5, Cmd_ReportTag & 0xff, (Cmd_ReportTag >> 8) & 0xff])
                        await client.write_gatt_char(PAR_WRITE_CHARACTERISTIC, params, response=True)
                        await asyncio.sleep(0.2)

                        # 查询参数确保设置成功
                        await client.write_gatt_char(PAR_WRITE_CHARACTERISTIC, bytes([0x19]), response=True)

                        # 保持运行直到断开连接
                        while client.is_connected:
                            await asyncio.sleep(1.0)

                    except Exception as e:
                        print(f"Error during BLE communication: {e}")
                    finally:
                        self.connection_status = "Disconnected"
                        print(f"Disconnected from {self.device_address}. Rescanning...")
                        with self.data_lock:
                            self.latest_data = {}  # 清空数据

            await asyncio.sleep(2)  # 断开连接后等待一会再重新扫描

    def _thread_target(self):
        """线程入口，为当前线程创建并运行asyncio事件循环"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_ble_loop())
        finally:
            loop.close()

    def start(self):
        """启动后台线程来处理BLE通信"""
        if self.ble_thread is None or not self.ble_thread.is_alive():
            print("Starting BLE background thread...")
            self.ble_thread = threading.Thread(target=self._thread_target, daemon=True)
            self.ble_thread.start()
            print("BLE background thread started.")

    def get_data(self):
        """获取最新的传感器数据 (线程安全)"""
        with self.data_lock:
            # 检查数据是否过时（超过2秒）
            if 'timestamp' in self.latest_data and (time.time() - self.latest_data['timestamp']) < 2.0:
                return self.latest_data.copy(), self.connection_status
            return {}, self.connection_status


# 创建一个单例供整个应用使用
sensor_manager = SensorManager()