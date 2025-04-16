# 팀장은 컬럼명, 파일명 정해야함
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
import time
import datetime



df_titles = pd.DataFrame()

options = ChromeOptions()

options.add_argument('lang=ko_KR') # 한국어를 사용하는 크롬 브라우저 옵션

service = ChromeService(executable_path=ChromeDriverManager().install()) # 크롬 드라이버 설치
driver = webdriver.Chrome(service=service, options=options)

titles = []

url = 'https://news.naver.com/section/102' # Social
driver.get(url)
time.sleep(5)

button_xpath = '//*[@id="newsct"]/div[4]/div/div[2]' # '기사 더보기' 버튼의 상대 경로(id 기준)
for i in range(15): # 15번 반복
    time.sleep(0.5) # 약간의 시간 딜레이
    driver.find_element(By.XPATH, button_xpath).click() # 설정한 경로의 버튼을 클릭
time.sleep(5) # 5초 기다리기

for i in range(1, 6): # 5회
    for j in range(1, 7): # 1부터 6까지
        time.sleep(0.5)
        # Xpath 예시_ 생활문화 부분
        '// *[ @ id = "newsct"] / div[4] / div / div[1] / div[1] / ul / li[1] / div / div / div[2] / a / strong'
        '// *[ @ id = "newsct"] / div[4] / div / div[1] / div[1] / ul / li[2] / div / div / div[2] / a / strong'
        '// *[ @ id = "newsct"] / div[4] / div / div[1] / div[1] / ul / li[3] / div / div / div[2] / a / strong'
        '//*[@id="newsct"]/div[4]/div/div[1]/div[1]/ul/li[4]/div/div/div[2]/a/strong'
        '//*[@id="newsct"]/div[4]/div/div[1]/div[1]/ul/li[4]/div/div/div[2]/a/strong'
        '//*[@id="newsct"]/div[4]/div/div[1]/div[1]/ul/li[5]/div/div/div[2]/a/strong'
        '//*[@id="newsct"]/div[4]/div/div[1]/div[1]/ul/li[6]/div/div/div[2]/a/strong'
        '//*[@id="newsct"]/div[4]/div/div[1]/div[2]/ul/li[1]/div/div/div[2]/a/strong'
        
        title_path = '//*[@id="newsct"]/div[4]/div/div[1]/div[{}]/ul/li[{}]/div/div/div[2]/a/strong'.format(i, j)
        try: # 해당 경로가 없을 수도 있으니까 예외 처리를 위한 try-except 문
            title = driver.find_element(By.XPATH, title_path).text # 요소 찾기.text
            print(title)
            titles.append(title)
        except:
            print('error', i, j)

df_social_titles = pd.DataFrame(titles, columns=['titles'])
df_social_titles['category'] = 'Social'
df_titles = pd.concat([df_titles, df_social_titles],
                          axis='rows', ignore_index=True)


titles = []

url = 'https://news.naver.com/section/103' # Culture
driver.get(url)
time.sleep(5)
button_xpath = '//*[@id="newsct"]/div[4]/div/div[2]/a' # '기사 더보기' 버튼의 상대 경로(id 기준)
for i in range(15): # 15번 반복
    time.sleep(0.5) # 약간의 시간 딜레이
    driver.find_element(By.XPATH, button_xpath).click() # 설정한 경로의 버튼을 클릭
time.sleep(5) # 5초 기다리기

for i in range(1, 6): # 5회
    for j in range(1, 7): # 1부터 6까지
        time.sleep(0.5)
        title_path = '//*[@id="newsct"]/div[4]/div/div[1]/div[{}]/ul/li[{}]/div/div/div[2]/a/strong'.format(i, j)
        try: # 해당 경로가 없을 수도 있으니까 예외 처리를 위한 try-except 문
            title = driver.find_element(By.XPATH, title_path).text # 요소 찾기.text
            print(title)
            titles.append(title)
        except:
            print('error', i, j)


df_culture_titles = pd.DataFrame(titles, columns=['titles'])
df_culture_titles['category'] = 'Culture'
df_titles = pd.concat([df_titles, df_culture_titles],
                          axis='rows', ignore_index=True)

#
df_titles.to_csv('{}_titles_Social_Culture.csv'.format(datetime.datetime.now().strftime('%Y%m%d')), index=False)
df_titles.info()