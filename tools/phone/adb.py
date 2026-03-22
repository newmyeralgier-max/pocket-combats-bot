"""
adb_wifi.py — автоматическое подключение adb по Wi‑Fi с ретраями и самодиагностикой.

Примеры:
  python adb_wifi.py --auto
  python adb_wifi.py --ip 192.168.0.100 --port 5555
  python adb_wifi.py --retries 5 --sleep 2
"""

import argparse
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import time
from typing import List, Optional, Tuple

ENC = "utf-8"


def log(msg: str):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "|", msg, flush=True)


def run(cmd: List[str], timeout: int = 15, capture: bool = True) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, shell=False)
        out = p.stdout.decode(ENC, errors="ignore")
        err = p.stderr.decode(ENC, errors="ignore")
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as e:
        return 1, "", str(e)


def adb(args: List[str], timeout: int = 15, serial: Optional[str] = None) -> Tuple[int, str, str]:
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += args
    return run(cmd, timeout=timeout)


def check_adb_available() -> bool:
    code, out, err = adb(["version"])
    if code == 0 and "Android Debug Bridge" in out:
        log("[adb] найден: " + out.strip().splitlines()[0])
        return True
    log("[adb] не найден или недоступен. Установите ADB и добавьте в PATH.")
    return False


def adb_kill_restart():
    log("[adb] перезапуск сервера…")
    adb(["kill-server"], timeout=10)
    adb(["start-server"], timeout=10)


def list_devices() -> List[Tuple[str, str]]:
    code, out, err = adb(["devices"])
    devs = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            devs.append((parts[0], parts[1]))
    return devs


def usb_device_serial() -> Optional[str]:
    for serial, state in list_devices():
        if ":" not in serial and state == "device":
            return serial
    return None


def is_ip_like(s: str) -> bool:
    return re.match("^\\d{1,3}(\\.\\d{1,3}){3}(:\\d{1,5})?$", s) is not None


def socket_probe(ip: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except Exception:
        return False


def parse_ip_from_output(txt: str) -> Optional[str]:
    m = re.search("\\bsrc\\s+(\\d{1,3}(?:\\.\\d{1,3}){3})\\b", txt)
    if m:
        return m.group(1)
    m = re.search("\\b(\\d{1,3}(?:\\.\\d{1,3}){3})\\b", txt)
    if m:
        return m.group(1)
    m = re.search("\\binet\\s+(\\d{1,3}(?:\\.\\d{1,3}){3})\\b", txt)
    if m:
        return m.group(1)
    m = re.search("(\\d{1,3}(?:\\.\\d{1,3}){3})", txt)
    if m:
        return m.group(1)
    return None


def get_device_ip(serial: str) -> Optional[str]:
    candidates = [
        ["shell", "ip", "route", "get", "1.1.1.1"],
        ["shell", "ip", "route"],
        ["shell", "ip", "-f", "inet", "addr", "show", "wlan0"],
        ["shell", "ifconfig", "wlan0"],
        ["shell", "getprop", "dhcp.wlan0.ipaddress"],
        ["shell", "getprop", "wlan0.ipaddress"],
    ]
    for args in candidates:
        code, out, err = adb(args, serial=serial)
        ip = parse_ip_from_output(out + " " + err)
        if ip and is_ip_like(ip):
            log(f"[wifi] найден IP устройства: {ip}")
            return ip
    log("[wifi] не удалось определить IP через ADB (проверьте Wi‑Fi на телефоне).")
    return None


def enable_tcpip(serial: str, port: int = 5555) -> bool:
    code, out, err = adb(["tcpip", str(port)], serial=serial)
    if code == 0 and ("restarting in" in out or "daemon not running" in out or "restart" in out.lower()):
        log(f"[wifi] tcpip:{port} — включено/перезапущено.")
        return True
    if code == 0 and out.strip():
        log(f"[wifi] tcpip ответ: {out.strip()}")
        return True
    if "error" in (out + err).lower():
        log(f"[wifi] ошибка tcpip: {out or err}")
    return code == 0


def adb_connect(ip: str, port: int = 5555) -> bool:
    target = f"{ip}:{port}"
    code, out, err = adb(["connect", target], timeout=10)
    text = (out + err).lower()
    if "connected to" in text or "already connected" in text:
        log(f"[wifi] подключено: {out.strip() or err.strip()}")
        return True
    if "connection refused" in text:
        log("[wifi] отказ в подключении — проверьте, включен ли ADB по TCP/IP (tcpip 5555).")
    elif "no route to host" in text:
        log("[wifi] нет маршрута — проверьте, в одной ли сети ПК и телефон.")
    elif "offline" in text:
        log("[wifi] устройство offline — попробую переподключиться.")
        adb(["disconnect", target])
    else:
        log(f"[wifi] не удалось подключиться: {out.strip() or err.strip()}")
    return False


def wait_for_device(target_serial: str, timeout_s: int = 12) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        for serial, state in list_devices():
            if serial == target_serial and state == "device":
                return True
        time.sleep(0.5)
    return False


def ensure_wifi(ip: Optional[str], port: int, retries: int, sleep_s: float) -> Optional[str]:
    """
    Возвращает serial вида ip:port при успехе, иначе None.
    Логика:
      1) Если задан IP — подключаемся напрямую (с проверкой сокета).
      2) Иначе ищем USB девайс, включаем tcpip, достаем IP и коннектимся.
    """
    if ip:
        if not socket_probe(ip, port):
            log(f"[net] порт {ip}:{port} недоступен (проверка сокета). Попробую через adb connect.")
        for i in range(retries):
            if adb_connect(ip, port):
                target = f"{ip}:{port}"
                if wait_for_device(target, timeout_s=6):
                    return target
            time.sleep(sleep_s)
        return None
    serial = usb_device_serial()
    if not serial:
        log("[usb] USB-устройство не найдено. Подключите телефон по USB или укажите --ip.")
        return None
    log(f"[usb] найдено устройство: {serial}")
    for s, state in list_devices():
        if s == serial and state != "device":
            log(f"[usb] состояние: {state}. Перезапуск ADB сервера…")
            adb_kill_restart()
            break
    if not enable_tcpip(serial, port=port):
        log("[wifi] не удалось включить tcpip. Попробуйте разблокировать экран и повторить.")
    ip_detected = get_device_ip(serial)
    if not ip_detected:
        log("[wifi] IP не определён. Проверьте Wi‑Fi на телефоне и сеть.")
        return None
    for i in range(retries):
        if adb_connect(ip_detected, port):
            target = f"{ip_detected}:{port}"
            if wait_for_device(target, timeout_s=6):
                return target
        time.sleep(sleep_s)
    return None


def disconnect_target(target: str):
    adb(["disconnect", target])
    log(f"[wifi] отключено: {target}")


def main():
    parser = argparse.ArgumentParser(description="Автоподключение ADB по Wi‑Fi с ретраями")
    parser.add_argument("--ip", type=str, help="IP телефона (если известен)")
    parser.add_argument("--port", type=int, default=5555, help="Порт ADB по TCP/IP (по умолчанию 5555)")
    parser.add_argument("--retries", type=int, default=4, help="Количество попыток подключения")
    parser.add_argument("--sleep", type=float, default=1.5, help="Пауза между попытками (сек)")
    parser.add_argument("--disconnect", action="store_true", help="Отключить ip:port и выйти")
    args = parser.parse_args()
    if not check_adb_available():
        sys.exit(1)
    target = f"{args.ip}:{args.port}" if args.ip else None
    if args.disconnect and target:
        disconnect_target(target)
        sys.exit(0)
    for serial, state in list_devices():
        if state == "offline" and ":" in serial:
            log(f"[cleanup] найден offline: {serial} — отключаю")
            adb(["disconnect", serial])
    connected = ensure_wifi(ip=args.ip, port=args.port, retries=args.retries, sleep_s=args.sleep)
    if connected:
        log(f"[ok] готово. Рабочее устройство: {connected}")
        code, out, err = adb(["shell", "getprop", "ro.product.model"], serial=connected, timeout=8)
        model = out.strip() or "unknown"
        log(f"[info] устройство: {model}")
        sys.exit(0)
    else:
        log("[fail] не удалось подключиться по Wi‑Fi.")
        hints = [
            "- Проверьте, что телефон и ПК в одной сети (одна подсеть).",
            "- Разблокируйте экран телефона и подтвердите дебаг по USB.",
            "- Выполните вручную: adb tcpip 5555; затем adb connect IP:5555.",
            "- Если не помогает — перезапустите ADB: adb kill-server; adb start-server.",
            "- Проверьте, не блокирует ли порт 5555 роутер/брандмауэр.",
        ]
        for h in hints:
            log("  " + h)
        sys.exit(2)


if __name__ == "__main__":
    main()
# python c:/bot/tools/phone/adb.py --ip 192.168.0.101 --port 5555
