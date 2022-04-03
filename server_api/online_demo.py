import json
import time
import requests


if __name__ == '__main__':

    data0 = {'sentence': "马龙赢了"}
    data1 = [
        {'sentence': "马龙赢了"},
        {'sentence': "地球有多大"}]
    data2 = "地球有多大"
    data3 = [{'sentence': "地球有多大"}]

    data = json.dumps(data1)  # data become str

    t = time.time()
    try:
        r = requests.get("http://127.0.0.1:8000/user/" + data)
        result = json.loads(r.text)  # from str to [dict, dict, dict, ...]
        print("type(result): ", type(result))
        print(result)
        print(json.dumps(result, sort_keys=False, indent=4, separators=(',', ':'), ensure_ascii=False))
    except:
        print('network instability, try again')
    print(time.time()-t)
