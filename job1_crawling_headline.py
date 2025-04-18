from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import datetime

category = ['Politiccs', 'Economic', 'Social', 'Culture', 'World', 'IT']
df_titles = pd.DataFrame()

# url = 'https://news.naver.com/section/100' # 네이버 뉴스의 [정치] 섹션 URL
# resp = requests.get(url) # 서버에 해당 URL 정보 요청 (HTTP의 GET 요청)
# soup = BeautifulSoup(resp.text, 'html.parser') # 응답받은 내용 정리
# print(list(soup))
#
# title_tags = soup.select('.sa_text_strong') # soup에서 sa_text_strong 클래스를 가진 것들만 선택
# print(list(title_tags))
#
# titles = []
# for tag in title_tags:
#     titles.append(tag.text)
# print(titles)
#
# df_section_titles = pd.DataFrame(titles, columns=['titles'])
# df_section_titles['category'] = category[0]
# print(df_section_titles.head())
# df_section_titles.info()

for i in range(6):
    url = 'https://news.naver.com/section/10{}'.format(i)  # 네이버 뉴스의 각 섹션 별 URL
    resp = requests.get(url)  # 서버에 해당 URL 정보 요청 (HTTP의 GET 요청)
    soup = BeautifulSoup(resp.text, 'html.parser')  # 응답받은 내용 정리
    print(list(soup))

    title_tags = soup.select('.sa_text_strong')  # soup에서 sa_text_strong 클래스를 가진 것들만 선택
    print(list(title_tags))

    titles = []
    for tag in title_tags:
        titles.append(tag.text)
    print(titles)

    df_section_titles = pd.DataFrame(titles, columns=['titles'])
    df_section_titles['category'] = category[i]
    df_titles = pd.concat([df_titles, df_section_titles],
                          axis='rows', ignore_index=True)
print(df_titles.head())
df_titles.info()
print(df_titles.category.value_counts())
df_titles.to_csv('./crawling_data/naver_headline_news_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index=False)
