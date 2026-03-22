import os
import subprocess
import time
from abc import ABC, abstractmethod
import numpy as np
import cv2
from typing import Optional

class ScreenCapture(ABC):
    @abstractmethod
    def grab(self) -> Optional[np.ndarray]:
        pass

class AdbCapture(ScreenCapture):
    def __init__(self, device_id: str, max_retries: int = 3, backoff_base: float = 0.5):
        self.device_id = device_id
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        
    def _adb_cmd(self, cmd: str, timeout: float = 10.0) -> bytes:
        args = ["adb"]
        if self.device_id:
            args.extend(["-s", self.device_id])
        args.extend(cmd.split())

        for attempt in range(self.max_retries):
            try:
                result = subprocess.run(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False
                )
                if result.returncode == 0:
                    return result.stdout
                
                print(f"[ADB] attempt {attempt+1} failed with code {result.returncode}: {result.stderr.decode(errors='ignore')}")
            except subprocess.TimeoutExpired:
                print(f"[ADB] attempt {attempt+1} timed out.")
                # Kill adb if timed out
                if attempt == self.max_retries - 1:
                    subprocess.run(["adb", "kill-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(["adb", "start-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(2)
            except Exception as e:
                print(f"[ADB] attempt {attempt+1} error: {e}")
                
            time.sleep(self.backoff_base * (2 ** attempt))
            
        return b""

    def grab(self) -> Optional[np.ndarray]:
        for attempt in range(max(1, self.max_retries)):
            raw = self._adb_cmd("exec-out screencap -p", timeout=15.0)
            if not raw:
                time.sleep(self.backoff_base)
                continue
                
            try:
                np_arr = np.frombuffer(raw, dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            except Exception as e:
                print(f"[AdbCapture] Failed to decode image: {e}")
                
            time.sleep(self.backoff_base)
            
        print("[AdbCapture] Could not grab valid screenshot via ADB after retries.")
        return None

class WindowCapture(ScreenCapture):
    def __init__(self, window_title: str):
        self.window_title = window_title
        import win32gui
        self.win32gui = win32gui
        self.hwnd = self._find_window(window_title)
        if not self.hwnd:
            print(f"[WindowCapture] CRITICAL: Window '{window_title}' not found. Capture will fail.")

    def _find_window(self, title: str) -> int:
        hwnd = self.win32gui.FindWindow(None, title)
        if not hwnd:
            # try to find by substring
            def callback(h, titles):
                t = self.win32gui.GetWindowText(h)
                if title.lower() in t.lower():
                    titles.append(h)
            found = []
            self.win32gui.EnumWindows(callback, found)
            if found:
                hwnd = found[0]
        return hwnd

    def grab(self) -> Optional[np.ndarray]:
        try:
            from mss import mss
        except ImportError:
            print("[WindowCapture] python-mss not installed. Using fallback (pywin32/cv2)...")
            return self._grab_win32()
            
        if not self.hwnd:
            self.hwnd = self._find_window(self.window_title)
            if not self.hwnd:
                return None
                
        # Get window rect
        try:
            left, top, right, bottom = self.win32gui.GetWindowRect(self.hwnd)
            width = right - left
            height = bottom - top
            if width <= 0 or height <= 0:
                return None
                
            monitor = {"top": top, "left": left, "width": width, "height": height}
            with mss() as sct:
                img = sct.grab(monitor)
                # Convert to numpy array in BGR format
                img_np = np.array(img)
                # mss returns BGRA, drop alpha
                if img_np.shape[2] == 4:
                    img_np = img_np[:, :, :3]
                return img_np
        except Exception as e:
            print(f"[WindowCapture] MSS grab failed: {e}")
            return self._grab_win32()
            
    def _grab_win32(self) -> Optional[np.ndarray]:
        try:
            import win32gui, win32ui, win32con
        except ImportError:
            print("[WindowCapture] pywin32 not installed!")
            return None
            
        if not self.hwnd:
            self.hwnd = self._find_window(self.window_title)
            if not self.hwnd:
                return None
                
        try:
            left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
            width = right - left
            height = bottom - top
            
            hwndDC = win32gui.GetWindowDC(self.hwnd)
            mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            
            saveDC.SelectObject(saveBitMap)
            
            result = win32gui.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 3) # 3: PW_RENDERFULLCONTENT
            
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            
            img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
            img = img[:, :, :3] # drop alpha
            
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwndDC)
            
            if result == 1:
                return img
            return None
        except Exception as e:
            print(f"[WindowCapture] win32 grab failed: {e}")
            return None

def create_capture(method: str, **kwargs) -> ScreenCapture:
    if method.lower() == "window":
        return WindowCapture(window_title=kwargs.get("window_title", "BlueStacks"))
    else:
        # Default to ADB
        return AdbCapture(
            device_id=kwargs.get("device_id", ""),
            max_retries=kwargs.get("max_retries", 3),
            backoff_base=kwargs.get("backoff_base", 0.5)
        )
