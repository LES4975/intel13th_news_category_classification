# 크롤링한 데이터 합치기

import pandas as pd
import glob

data_dir = './crawling_data/titles_dataset/'
data_path = glob.glob(data_dir + '*.*')
print(data_path)

df = pd.DataFrame() # 빈 데이터프레임 만들기
for path in data_path: # 모든 파일에 대하여
    df_section = pd.read_csv(path)
    df = pd.concat([df, df_section], ignore_index=True) # 수집한 각기 데이터를 합치기
df.info()
print(df.head())
df.to_csv('./crawling_data/news_titles.csv')