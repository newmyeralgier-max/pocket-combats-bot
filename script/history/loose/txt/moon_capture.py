import subprocess


def capture_screen(serial, out_path):
    with open(out_path, "wb") as f:
        subprocess.run(["adb", "-s", serial, "exec-out", "screencap", "-p"], stdout=f)


capture_screen("192.168.0.100:5555", "screen.png")
print("✅ Скриншот сохранён: screen.png")
