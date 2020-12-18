# ライブラリ
import os
from dotenv import load_dotenv
from requests_oauthlib import OAuth1Session
import oauth2 as oauth
import sys
import json
import glob
import pandas as pd
import time
import numpy as np
import joblib
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from janome.tokenizer import Tokenizer
from gensim.models import word2vec
import ast

###################################################################################
#データ収集の関数
###################################################################################
#TwitterAPIを使用するための情報を保存する関数
def define_client_proc():
    dotenv_path = '.env'
    load_dotenv(dotenv_path)
    CONSUMER_KEY = os.environ.get("CONSUMER_KEY")
    CONSUMER_SECRET = os.environ.get("CONSUMER_SECRET")
    ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")
    ACCESS_TOKEN_SECRET = os.environ.get("ACCESS_TOKEN_SECRET")
    consumer = oauth.Consumer(key=CONSUMER_KEY, secret=CONSUMER_SECRET)
    access_token = oauth.Token(key=ACCESS_TOKEN, secret=ACCESS_TOKEN_SECRET)
    client = oauth.Client(consumer, access_token)
    return client

#
def get_limit():
    url = 'https://api.twitter.com/1.1/application/rate_limit_status.json'
    return define_client_proc().request(url)

# Twitterから情報を取得する関数------------------------------------------------
def get_tweets_proc(client, user_id, max_get_tweets_count):
    url_base = "https://api.twitter.com/1.1/statuses/user_timeline.json?user_id="
    url = url_base + user_id + "&count=" + str(max_get_tweets_count)
    array_tweets_from_json = []
    response, data = client.request(url)
    if response.status == 200:
        json_str = data.decode('utf-8')
        array_tweets_from_json = json.loads(json_str)
        sys.stderr.write("len(array_tweets_from_json) = %d\n" % len(array_tweets_from_json))
    else:
        sys.stderr.write("*** error *** get_ids_proc ***\n")
        sys.stderr.write("Error: %d\n" % response.status)
    return  array_tweets_from_json


###################################################################################

# idの履歴を取得する関数
def read_id():
    if os.path.exists('id.txt'):
        with open("id.txt", mode="r") as f:
            return f.readlines()
    else :
        with open("id.txt", mode="w") as f:            
            f.write("")
    return []

############################################################

def make_user_dataflame(user_id, max_get_tweets_count):
    client = define_client_proc()
    array_tweets_from_json = get_tweets_proc(client,user_id, max_get_tweets_count)

    # 取得してきた人がすでに蓄積されているかを判定する
    new_person = False
    if array_tweets_from_json[0]['user']['id_str'] + '\n' in read_id():
        pass
    else :
        new_person = True
        with open("id.txt", mode = "a") as f:
            f.write(array_tweets_from_json[0]['user']['id_str'] + '\n')

    # ユーザーデータフレームの読み込み
    if not os.path.exists('user.csv'):
        df = pd.DataFrame(
            columns=["id_str", "name", "introduction", "created_at"]
        )
        df.to_csv('user.csv', index = False)
    df = pd.read_csv('user.csv')

    # 取得してきた情報が新規だった場合ユーザー情報を書き込む
    id_str = array_tweets_from_json[0]['user']['id_str']
    if new_person:
        name = array_tweets_from_json[0]['user']['name']
        description = array_tweets_from_json[0]['user']['description']
        created_at = array_tweets_from_json[0]['user']['created_at']
        add_df = pd.DataFrame()
        add_df['id_str'] = [id_str]
        add_df['name'] = [name]
        add_df['introduction'] = [description]
        add_df['created_at'] = [created_at]
        #dfとadd_dfの連結
        df = pd.concat([df, add_df])
        df.to_csv('user.csv', index = False)

    # ユーザーのアカウント情報、ツイート情報をデータフレームとして追加する
    if new_person:
        id_df = pd.DataFrame()
        if not os.path.exists('user'):
            os.mkdir('user')
        if not os.path.exists(f'user/{id_str}.csv'):
            df = pd.DataFrame(
                columns=[
                    "text", "tweeted_at", "retweet_count", "favorite_count",
                    "hashtags", "symbols", "tweet_count", "friends_count", 
                    "followers_count"
                ]
            )
            df.to_csv(f'user/{id_str}.csv', index = False)
    else:
        id_df = pd.read_csv(f'user/{id_str}.csv')

    text, retweet_count, favorite_count, created_at, hashtags, symbols, statuses_count, friends_count, followers_count = [], [], [], [], [], [], [], [], []
    for tweet in array_tweets_from_json:
        if new_person or (not tweet['created_at'] in id_df['tweeted_at'].values):
            text.append(tweet['text'])
            retweet_count.append(tweet['retweet_count'])
            favorite_count.append(tweet['favorite_count'])
            created_at.append(tweet['created_at'])
            hashtags.append(tweet['entities']['hashtags'])
            symbols.append(tweet['entities']['symbols'])
            statuses_count.append(tweet['user']['statuses_count'])
            friends_count.append(tweet['user']['friends_count'])
            followers_count.append(tweet['user']['followers_count'])

    add_id_df = pd.DataFrame()
    add_id_df['text'] = text
    add_id_df['tweeted_at'] = created_at
    add_id_df['retweet_count'] = retweet_count
    add_id_df['favorite_count'] = favorite_count
    add_id_df['hashtags'] = hashtags
    add_id_df['symbols'] = symbols
    add_id_df['tweet_count'] = statuses_count
    add_id_df['friends_count'] = friends_count
    add_id_df['followers_count'] = followers_count
    id_df = pd.concat([id_df, add_id_df])
    id_df.to_csv(f'user/{id_str}.csv', index = False)

    sys.stderr.write("len(array_tweets_from_json) = %d\n" % len(array_tweets_from_json))
    sys.stderr.write("*** 終了 ***\n")


###########################################################

# 欲しいユーザーのデータを自動取得してデータフレームに加える関数
def autosave_user_data(user_id, max_get_tweets_count):
    client = define_client_proc()
    array_tweets_from_json = get_tweets_proc(client, user_id, max_get_tweets_count)
    with open("id.txt", mode = "r") as f:
        f.readlines(array_tweets_from_json[0]['user']['id_str'] + '\n')
    user_id_list = []
    for i in user_id_list:
        make_user_dataflame(i, 1000)
        time.sleep(10)
        if int(limit[0]['x-rate-limit-remaining']) <= 0 :
            f = True
            while f:
                time.sleep(60)
                if int(limit[0]['x-rate-limit-reset']) - time.time() < 0:
                    f = False
                    limit = get_limit()


###################################################################################
#自身のユーザーIDを取得する関数
def read_my_id():
    url = 'https://api.twitter.com/1.1/account/verify_credentials.json'
    user_object = define_client_proc().request(url, 'GET')
    user_object_txt = user_object[1].decode('utf-8')
    user_object_dict = ast.literal_eval(user_object_txt.replace('false', 'False').replace('true', 'True').replace('null', '0'))
    return user_object_dict['id_str']

###################################################################################
#自分のツイートデータを取得する関数
def get_own_tweet_data():
    # ユーザーの情報
    df = pd.read_csv(f'user/{read_my_id()}.csv')
    # ユーザーのツイート情報
    user_tweet = df['text'].values
    return user_tweet

###################################################################################
#全員分のデータを取得する関数
def get_all_user_data():
    user_csv_lst = glob.glob('user/*.csv')
    df = pd.DataFrame()
    for i in user_csv_lst:
        df = pd.concat([df, pd.read_csv(i)])
    return df

###################################################################################
# 分析
###################################################################################
#自分と他のユーザーのツイートのベクトルを取得する関数
def make_user_vec(loaded_clf):
    tokenizer = Tokenizer()
    user_vec_lst = []
    user_id_lst = glob.glob('user/*.csv')
    my_id = read_my_id()
    del user_id_lst[user_id_lst.index(f'user/{my_id}.csv')]
    for n in range(len(user_id_lst) + 1):
        # print(f'{n} / {len(user_id_lst) + 1}')
        # ユーザーの情報
        if n == 0:
            df = pd.read_csv(f'user/{my_id}.csv')
        else :
            df = pd.read_csv(user_id_lst[n - 1])
        # ユーザーのツイート情報
        user_tweet = df['text'].values
        new_user_tweet = []
        for i in list(map(lambda x: x.split('https')[0], user_tweet)):
            if len(i) >= 1 and '@' == i[0]:
                new_user_tweet.append(i.split(' ')[1])
            else:
                new_user_tweet.append(i)
        sentences = new_user_tweet
        words = []
        for s in sentences:
            for i in tokenizer.tokenize(s):
                if i.part_of_speech.startswith('名詞'):
                    if len(i.surface) >= 2:
                        if i.surface.isdigit() == False:
                            words.append(i.surface)
        vector_lst = []
        for i in words:
            if i in loaded_clf.wv.vocab:
                vector = loaded_clf.wv[i]
                word = loaded_clf.wv.most_similar([ vector ], [], 10)
                for j in word:
                    vector_lst.append(loaded_clf.wv[j[0]]*j[1])
        sum_vec = [0 for _ in range(100)]
        for i in vector_lst:
            for j in range(len(i)):
                sum_vec[j] += i[j]
        user_vec_lst.append(list(map(lambda x: x/len(vector_lst), sum_vec)))
    return user_vec_lst

###################################################################################
#２つのベクトルの距離を測る関数
def cos_sim(v1, v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

###################################################################################
#自分と類似しているユーザーの名前を出力する
def your_similar_user(user_id_lst):
    # モデルの読み込み
    loaded_clf = joblib.load('model.sav')
    flag = int(input('簡易実行にしますか？\nはい：1\nいいえ：0'))
    if flag:
        user_vec_lst = joblib.load('user_vec_lst.sav')
    else :
        user_vec_lst = make_user_vec(loaded_clf)
    cos_dis = []
    for i in range(1, len(user_vec_lst)):
        cos_dis.append(cos_sim(user_vec_lst[0],user_vec_lst[i]))
    user_df = pd.read_csv('user.csv')
    similar_user = user_df[user_df['id_str'] == int(user_id_lst[cos_dis.index(max(cos_dis)) + 1].split('/')[1].split('.')[0])]['name'].values[0]
    return similar_user



###################################################################################

#関数をデータ収集と分析に分ける
#ギットハブにあげる
#リードミーを書く
#ファイル分けは後

#####################################################################################

# def save_user_data():
#     with open("id.txt", mode = "r") as f:
#         user_id_list = list(map(lambda x: x.split('\n')[0], f.readlines()))
#     #user_id_list = ['935676428971941889', '2793976860','988234323680677888', '148722233', '3314978227', '4005258913', '1195700876', '570651448', '1027842833875525633', '738199700914872320', '2346263425', '3649925052','744469005730144257', '2601085537', '221713390', '224308823', '227540020', '255356284', '515315620', '1003250335', '1616729371', '798164171875438592','1335296648', '321424506', '518608419', '634586690', '967629192', '5865072', '725283168899592194', '128202499', '784115819827519488', '3315111726']
#     limit = get_limit()
#     for i in user_id_list:
#         main(i, 1000)
#         time.sleep(0.1)
#         if int(limit[0]['x-rate-limit-remaining']) <= 0 :
#             f = True
#             while f:
#                 time.sleep(0.1)
#                 if int(limit[0]['x-rate-limit-reset']) - time.time() < 0:
#                     f = False
#                     limit = get_limit()

###################################################################################
# 特定のユーザーの情報を抽出
# def get_select_user_data():
#     from IPython.display import clear_output
#     # with open('id.txt') as f:
#     #     id_lst = list(map(lambda x: x.split('\n')[0], f.readlines()))
#     df = pd.read_csv('user.csv')
#     print('誰の情報を取得したいですか？\n番号を入力してください。\n-------------------')
#     id_name = list(zip(df.index, df['name'].values))
#     for i, j in id_name:
#         print(i, j)
#     idn = int(input('番号を入力 : '))
#     clear_output()
#     print(f'「{id_name[idn][1]}」を取得します。')
#     df = pd.read_csv('user/' + str(df[df['name'] == id_name[idn][1]]['id_str'].values[0]) + '.csv')
#     return df