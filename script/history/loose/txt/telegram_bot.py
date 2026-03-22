import requests

TOKEN = "AAHZGVOFziKmsOnzTb6Tt6Kx0h-OBo2tAXg"
CHAT_ID = "7988165585"


def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": text})
