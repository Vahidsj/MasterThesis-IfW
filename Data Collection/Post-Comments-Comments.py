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
post_comments_id = []
post_comments_comments_message = []
post_comments_comments_from_name = []
post_comments_comments_from_id = []
post_comments_comments_created_time = []

table = Table.read_table('C:/Users/Vahid/Desktop/...')
page_posts_id = table.column(0)
comments_id = table.column(1)

page_posts_comments = []
for i in np.arange(0, len(page_posts_id)):
    
    if comments_id[i] == "No Comment for Post":
        page_posts_comments.append("No Comment for Post")
    else:
        page_posts_comments.append(page_posts_id[i].split('_')[0] + '_' + comments_id[i])

def comment_comment_query(comment):
    
    page_posts_comments_id.append(str(p_p_c_id))
    post_comments_id.append(str(comment['id']))
    post_comments_comments_created_time.append(comment['created_time'])
    post_comments_comments_from_name.append(str(comment['from']['name']))
    post_comments_comments_from_id.append(str(comment['from']['id'])) # Here is a Problem - this String here is not the same as the actual USER ID is!!
    post_comments_comments_message.append(str(comment['message']))
    
    print(str(comment['from']['id'])) # Here is a Problem - this String here is not the same as the actual USER ID is!!
        
    return (page_posts_comments_id, post_comments_id, post_comments_comments_created_time, post_comments_comments_from_name, post_comments_comments_from_id, post_comments_comments_message)


for p_p_c_id in page_posts_comments:

    if p_p_c_id == "No Comment for Post":
        
        page_posts_comments_id.append('No Comment for Post')
        post_comments_id.append('No Comment for Comment')
        post_comments_comments_created_time.append('-')
        post_comments_comments_from_name.append('-')
        post_comments_comments_from_id.append('-')
        post_comments_comments_message.append('No Message for CC')
        
    else:
        
        comments_comments = graph.get_connections(id=str(p_p_c_id), connection_name='comments')
        
        if not comments_comments['data']:
            # some comments have no comments
            page_posts_comments_id.append(str(p_p_c_id))
            post_comments_id.append('No Comment for Comment')
            post_comments_comments_created_time.append('-')
            post_comments_comments_from_name.append('-')
            post_comments_comments_from_id.append('-')
            post_comments_comments_message.append('No Message for CC')
            print('No Comment',p_p_c_id)
        else:
            while True:
                try:
                    # Perform some action on each post in the collection we receive from
                    # Facebook.
                    [comment_comment_query(comment=comment) for comment in comments_comments['data']]
                    # Attempt to make a request to the next page of data, if it exists.
                    comments_comments = requests.get(comments_comments['paging']['next']).json()
                except KeyError:
                    print('No More Page for Likes!')
                    # When there are no more pages (['paging']['next']), break from the
                    # loop and end the script.
                    break

HamburgPosts_comments_comments = Table().with_columns('Page-Post-Comment ID', page_posts_comments_id,
                                            'Post-Comment ID', post_comments_id,
                                            'Comment-CreatedTime' , post_comments_comments_created_time,
                                            'Comment-User ID', post_comments_comments_from_id,
                                            'Comment-User Name', post_comments_comments_from_name,
                                            'Comment-Comment Message', post_comments_comments_message)

HamburgPosts_comments_comments.to_csv('HamburgPosts_comments_comments.csv')
