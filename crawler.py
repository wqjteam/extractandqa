# 创 建 人:Cui.
# 创建时间:2021/8/30 16:45

# 'http://www.ihchina.cn/project#target1'

"""http://www.ihchina.cn/Article/Index/getProject.html
 ?province=&rx_time=&type=&cate=&keywords=&category_id=16&limit=10&p=页数"""

# 引用库
import time

import requests
import json
import csv

# 准备.csv
f = open('data/origin/国家级非物质文化遗产代表性项目名录.json', mode='w', encoding='utf-8', newline='')
# writer = csv.writer(f)
# 写入表头
# writer.writerow(["名称", "公布时间", "申报地区或单位", "编号", "类型", "详细信息"])

# 请求头
h = {
    "Host": "www.ihchina.cn",
    "Referer": "http://www.ihchina.cn/project",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/92.0.4515.159 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest"
}

num_set=set()

def get_info(num):
    # 空参数可以删掉
    url = f"http://www.ihchina.cn/Article/Index/getProject.html?category_id=16&limit=10&p={num}"
    resp = requests.get(url, headers=h)

    # 当时解析json的时候导出了一下
    # with open('info362.json', 'w') as f:
    #     f.write(str(resp.json()))
    JsonObj=None
    try:
        JsonObj = json.loads(resp.text)
    except TypeError as e:
        print(e)
    try:
        count = len(JsonObj["list"])
    except TypeError as e:
        return 'error'
    for i in range(count):
        jsondict = {}
        # 名称
        name = JsonObj["list"][i]["title"]
        jsondict['name'] = name
        # 公布时间
        gb_time = JsonObj["list"][i]["rx_time"]
        # 申报地区或单位
        gb_time = gb_time.replace('</br>', '')
        jsondict['gb_time'] = gb_time

        province = JsonObj["list"][i]["province"]
        jsondict['province'] = province
        # 编号
        num = JsonObj["list"][i]["num"]
        jsondict['num'] = num
        # 类型
        cate = JsonObj["list"][i]["cate"]
        jsondict['cate'] = cate
        # 详细信息
        content = JsonObj["list"][i]["content"]
        content = content.replace('&lt;br /&gt;\r\n\u3000\u3000', '')
        jsondict['content'] = content


        if num not in num_set and len(content) >=1:
            # 写入文件中
            time.sleep(1)
            json.dump(jsondict, f, ensure_ascii=False)
            # json.dump(jsondict, f, ensure_ascii=False, indent=1)
            #换行
            f.write("\n")
            num_set.add(num)
            # f.writelines()
            # writer.writerow([name, gb_time, province, num, cate, content])


if __name__ == '__main__':
    i = 1
    # 到最后一页后自动跳出
    while True:
        if get_info(i) != 'error':
            print(i, 'ok')
        else:
            break
        i += 1
    f.close()

    print('全部ok')
