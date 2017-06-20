import facebook
import requests
import numpy as np
import sys
from datascience import *

# You should paste your access_token here. You need the short lived access token , because we don't have version 2.3 of the long lived access token. post_query function works faster with this version. >>> See Instruction 1.
access_token = 'YOUR ACCESS TOKEN'
graph = facebook.GraphAPI(access_token, version='2.3')

#outputs
page_posts_comments_id = []
post_comments_likes_name = []
post_comments_likes_id = []

table = Table.read_table('C:/Users\Vahid\Desktop/...')
page_posts_id = table.column(0)
comments_id = table.column(1)

page_posts_comments = []
for i in np.arange(0, len(page_posts_id)):
    
    if comments_id[i] == "No Comment for Post":
        page_posts_comments.append("No Comment for Post")
    else:
        page_posts_comments.append(page_posts_id[i].split('_')[0] + '_' + comments_id[i])

def comment_like_query(like):
    
    page_posts_comments_id.append(str(p_p_c_id))
    post_comments_likes_name.append(str(like['name']))
    post_comments_likes_id.append(str(like['id']))
    
    print(str(like['id']))
        
    return (page_posts_comments_id,post_comments_likes_name, post_comments_likes_id)


for p_p_c_id in page_posts_comments:

    if p_p_c_id == "No Comment for Post":
        
        page_posts_comments_id.append('No Comment for Post')
        post_comments_likes_name.append('-')
        post_comments_likes_id.append('-')
        
    else:
        
        comments_likes = graph.get_connections(id=str(p_p_c_id), connection_name='likes')
        
        if not comments_likes['data']:
            # some comments have no likes
            page_posts_comments_id.append(str(p_p_c_id))
            post_comments_likes_name.append('-')
            post_comments_likes_id.append('-')
            print('No Comment',p_p_c_id)
            
        else:
            while True:
                try:
                    # Perform some action on each post in the collection we receive from
                    # Facebook.
                    [comment_like_query(like=like) for like in comments_likes['data']]
                    # Attempt to make a request to the next page of data, if it exists.
                    comments_likes = requests.get(comments_likes['paging']['next']).json()
                except KeyError:
                    print('No More Page for Comments!')
                    # When there are no more pages (['paging']['next']), break from the
                    # loop and end the script.
                    break

HamburgPosts_comments_likes = Table().with_columns('Page-Post-Comment ID', page_posts_comments_id,
                                            'Post-Comment-Like-User Name', post_comments_likes_name,
                                            'Post-Comment-Like-User ID' , post_comments_likes_id)

HamburgPosts_comments_likes.to_csv('HamburgPosts_comments_likes.csv')
