"""
@ Author: wyfffffei
@ Project: Expressway-Detect-Yolov5
@ File: crawler.py
@ Time: 2022/3/26 18:02
@ Function:  
"""
import time
import requests
import shutil
# from pyquery import PyQuery as pq


# proxies = {"socks5": "socks5://127.0.0.1:1080", "http": "http://127.0.0.1:1081"}
agent = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36"}

# for page in range(1, 11):
#     url = "https://www.gettyimages.com/photos/throwing-garbage-on-road?assettype=image&sort=mostpopular&phrase=throwing%20garbage%20on%20road&license=rf,rm&page={}".format(page)
#     html = requests.get(url, headers=agent)
#     if html.status_code == 200:
#         html.encoding = "utf-8"
#         html = pq(html.text)
#         items = html.find("source")
#         for item in items.items():
#             pic_url = item.attr("srcset")
#             file = open("./output.txt", 'a+')
#             file.writelines(pic_url + '\n')
#             file.close()

with open("./output_2.txt", 'r') as data:
    data = data.readlines()
    for i, item in enumerate(data):
        # url = "https://media.gettyimages.com/photos/woman-separating-waste-and-pet-bottle-for-waste-management-picture-id1255595409?s=612x612"
        url = item.strip()
        pic = requests.get(url, stream=True)
        with open('./output/car_garbage_{:0>3d}.png'.format(i), 'wb') as out_file:
            shutil.copyfileobj(pic.raw, out_file)
        del pic
        time.sleep(1)

