from datascience import *
import facebook
import requests
import urllib
from urllib import request
import os

table = Table.read_table('C:/Users\Vahid\Desktop/UserID.csv')
user_ids_modified = table.column(0)

user_ids = []
user_ids = [str(i[1:len(i)-1]) for i in user_ids_modified]

# change the directory
os.chdir("D:/Profile Pictures - Facebook")

access_token = 'EAAI6JhLteM8BAEEXVabHGgVZBjEZBRDOce8ZCVj0nWWG5ibDBkQR0B4KZABNXDXKfhlGAcvOZC9NuyHaZBZBgRSQwLYNO5lnZBrJZCWQWbIK82LTgWqZAWbLwe8ZCm6afEwCNC6LbjlemtTIGPMeYhBzFbUJZBpkZCsNxZCgUZD'
graph = facebook.GraphAPI(access_token, version='2.5')

for user_id in user_ids:
    
    if user_id == '-':
        print(user_id)
    else:
        comments = graph.get_connections(id=str(user_id), connection_name='picture')
        url = comments['url']
        urllib.request.urlretrieve(str(url), str(user_id)+'.jpg')
        print(user_id)
