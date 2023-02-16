import requests

def send_telegram(message):
    TOKEN = "6201723110:AAFh6na0r7FVA44Ox4c7pUd2Gzb9Xl3kbKk"
    chat_id = "6191158308"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    resp = requests.get(url).json()
    return resp
    
def send_photos(file):
    token = "6201723110:AAFh6na0r7FVA44Ox4c7pUd2Gzb9Xl3kbKk"
    chat_id = "6191158308"
    method = "sendPhoto"
    params = {'chat_id': chat_id}
    files = {'photo': file}
    url = f"https://api.telegram.org/bot{token}/"
    resp = requests.post(url + method, params, files=files)
    
    return resp

if __name__ == '__main__':
    message = "hello from your telegram bot"
    send_telegram(message)
    send_photos(open(r"D:\MurrayBrent\projects\TR3D\checkpoints\dgcnn_pointaugment_4096\output\confusion_matrix.png", 'rb'))