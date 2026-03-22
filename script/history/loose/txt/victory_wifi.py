import os
import subprocess

DEVICE_IP = "192.168.0.100:5555"
DEBUG_PATH = "debug"
SCREENSHOT = os.path.join(DEBUG_PATH, "last.png")


def connect_device(ip):
    print(f"[wifi] подключение к {ip}...")
    subprocess.run(["adb", "connect", ip])


def capture_screen(ip, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[screen] снимаем экран → {out_path}")
    with open(out_path, "wb") as f:
        subprocess.run(["adb", "-s", ip, "exec-out", "screencap", "-p"], stdout=f)


def main():
    connect_device(DEVICE_IP)
    capture_screen(DEVICE_IP, SCREENSHOT)
    print("✅ Готово! Скрин в debug/last.png")


if __name__ == "__main__":
    main()
