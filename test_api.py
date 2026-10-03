import requests
import json

url = 'http://localhost:8000/meetings/process'
print("Sending POST request to", url)
try:
    with open('C:\\MoM Fresh\\sample3.wav', 'rb') as f:
        files = {'audio_file': f}
        response = requests.post(url, files=files)
        
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        print("Success!")
        data = response.json()
        print("MOM Output:")
        print(json.dumps(data.get('mom'), indent=2))
    else:
        print("Error:", response.text)
except Exception as e:
    print("Request failed:", e)
