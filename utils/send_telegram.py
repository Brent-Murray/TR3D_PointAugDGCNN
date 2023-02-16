import requests

def send_telegram(message, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}"
    resp = requests.get(url).json()
    return resp
    
def send_photos(file, token, chat_id):
    method = "sendPhoto"
    params = {'chat_id': chat_id}
    files = {'photo': file}
    url = f"https://api.telegram.org/bot{token}/"
    resp = requests.post(url + method, params, files=files)
    
    return resp
