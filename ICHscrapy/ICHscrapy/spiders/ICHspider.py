# -*- coding: utf-8 -*-
import time

import scrapy
import pymysql
from ICHscrapy.items import IchscrapyItem


class ICHspider(scrapy.Spider):
    name = "ich_data_sources"
    allowed_domains = ["www.ihchina.cn"]
    start_urls = ['https://www.ihchina.cn/project.html#target1']

    def start_requests(self):
        yield scrapy.Request(url=ICHspider.start_urls[0], callback=self.parse, meta={'chrome_flag': 1},
                             dont_filter=True)

    def parse(self, response):

        lis = response.xpath("//div[@id='page']/ul/li[@id='down']")
        for li in lis:
            type = li.xpath("./h2/a[1]/span/text()").extract()[0].replace("[", "").replace("]", "")
            title = li.xpath("./h2/a[2]/text()").extract()[0]
            contenturl = li.xpath("./h2/a[2]/@href").extract()[0]
            # 可以使用该方法在此直接获取文章信息
            # contentResponse = requests.get(contenturl)
            # contentResponse.encoding = 'utf-8'
            # print(contentResponse.text)
            # contentResponse_2 = etree.HTML(contentResponse.text)
            # paragraphs = contentResponse_2.xpath("//div[@class='qh_en']/p/text()")
            # print(paragraphs)

            # 判断该url是否已经存在库中,存在库中,则不进行爬取
            if (contenturl.strip() not in self.urlSet):
                yield scrapy.Request(response.urljoin(contenturl), callback=self.contentParse,
                                     meta={'chrome_flag': 0, "type": type, "title": title})
            else:
                self.repeat_url_number = self.repeat_url_number + 1
                if (self.repeat_url_number > 100):
                    self.crawler.engine.close_spider(self, "重复次数过多,超过100次")

        next_href = response.xpath("//div[@class='page th']/a[text()='下一页']/@href").extract()
        self.log('Saved file %s.')  # self.log是运行日志，不是必要的
        if next_href is not None and len(next_href):  # 判断是否存在下一页
            next_page = response.urljoin(next_href[0])
        yield scrapy.Request(next_page, callback=self.parse)

    def contentParse(self, response):
        ich = IchscrapyItem()
        ich['theme'] = response.meta['type']
        ich['title'] = response.meta['title']
        ich['date'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        # 获取所有的段落
        paragraphs = response.xpath("//div[@class='qh_en']/p/text()").extract()

        ich['content'] = "|||".join(paragraphs)
        ich['paragraphs'] = len(paragraphs)
        ich['claw_url'] = response.url.strip()
        if (ich['content'] is None or ich['content'] == ""):
            pass
        else:
            yield ich

    def __init__(self):
        self.urlSet=()
        # # 设置全局的重复url统计次数
        # self.repeat_url_number = 0;
        # # 建立连接
        # self.conn = pymysql.connect(host='localhost', user='root', passwd='buaaai123456', port=3306,
        #                             db='AIcourse')  # 有中文要存入数据库的话要加charset='utf8'
        # # 创建游标
        # self.cursor = self.conn.cursor()
        # select_sql = """
        #                 select claw_url from  Economist
        #                 """
        # self.cursor.execute(select_sql)
        # results = self.cursor.fetchall()
        # self.urlSet = set(str(url[0]).strip() for url in results)
        # # urlSet=set(hash(url[0]).strip for url in results)
        # self.cursor.close()
        # self.conn.close()
