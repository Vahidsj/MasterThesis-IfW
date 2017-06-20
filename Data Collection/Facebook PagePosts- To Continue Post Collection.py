import pandas as pd
import facebook
import requests
from datascience import *
import smtplib
import os
import pickle

# You should paste your access_token here. You need the short lived access token , because we don't have version 2.3 of the long lived access token. post_query function works faster with this version. >>> See Instruction 1.
access_token = 'Your Access Token'
    

all_done_FileNames = []
counter = 0

# Save done pages to pretend to repeat the process unnecessary if only the Internet disconnected
try:
    with open('all_done_FileNames.pkl', 'rb') as f:
        all_done_FileNames = pickle.load(f)
except:
    pass

# Paste the file address of CSV files
file_address = str('C:/Users/Vahid/Desktop/.../...')
for filename in os.listdir(str(file_address)):

    if filename not in all_done_FileNames:
    
        user = str(filename.split("-")[0])

        newest_PostID = [Table.read_table(str(file_address)+ "/" + str(filename), encoding="ISO-8859-1").column(0)[0],
                         Table.read_table(str(file_address)+ "/"  + str(filename), encoding="ISO-8859-1").column(0)[1],
                         Table.read_table(str(file_address)+ "/"  + str(filename), encoding="ISO-8859-1").column(0)[2]]
        
        # Outputs
        post_id = []
        post_time = []
        post_type = []
        post_message = []
        post_attachment_title = []
        post_attachment_description = []
        post_attachment_link = []

        # ID query
        def id_query(post):
            return(post['id'])

        # Collecting the post id, post created time, post type, and post message (post description) etc.
        def post_query(post):
            
            global counter
            counter = counter+1

            # It's a checker to request you a new access_token because of expiration time. You can increase/decrease this number based on your Internet Speed.
            if counter < 40000:
                
                print(counter)

                # some posts have no links
                if 'link' in post:
                    post_attachment_link.append(post['link'])
                    print(post['created_time'])
                    
                    # some posts have no post messages (post description)
                    if 'message' in post and 'name' in post:
                        
                        post_id.append(post['id'])
                        post_time.append(post['created_time']) # The created Time is according to GMT!
                        post_type.append(post['type'])
                        post_message.append(post['message'])
                        post_attachment_title.append(post['name'])

                    elif 'message' in post and 'name' not in post:
                        
                        post_id.append(post['id'])
                        post_time.append(post['created_time']) # The created Time is according to GMT!
                        post_type.append(post['type'])
                        post_message.append(post['message'])
                        post_attachment_title.append('No Title for this Attachment')

                    elif 'message' not in post and 'name' in post:

                        post_id.append(post['id'])
                        post_time.append(post['created_time']) # The created Time is according to GMT!
                        post_type.append(post['type'])
                        post_message.append('No Message for this Post')
                        post_attachment_title.append(post['name'])
                    
                    else:
                         
                        post_id.append(post['id'])
                        post_time.append(post['created_time'])
                        post_type.append(post['type'])
                        post_message.append('No Message for this Post')
                        post_attachment_title.append('No Title for this Attachment')

                    if 'description' in post:
                        
                        post_attachment_description.append(post['description'])

                    else:

                        post_attachment_description.append('No Description for this Attachment')
                        
                else:
                    
                    post_attachment_link.append('No Link for this Post')
                    print(post['created_time'])
                    
                    if 'message' in post and 'name' in post:
                            
                        post_id.append(post['id'])
                        post_time.append(post['created_time']) # The created Time is according to GMT!
                        post_type.append(post['type'])
                        post_message.append(post['message'])
                        post_attachment_title.append(post['name'])

                    elif 'message' in post and 'name' not in post:
                        
                        post_id.append(post['id'])
                        post_time.append(post['created_time']) # The created Time is according to GMT!
                        post_type.append(post['type'])
                        post_message.append(post['message'])
                        post_attachment_title.append('No Title for this Attachment')

                    elif 'message' not in post and 'name' in post:

                        post_id.append(post['id'])
                        post_time.append(post['created_time']) # The created Time is according to GMT!
                        post_type.append(post['type'])
                        post_message.append('No Message for this Post')
                        post_attachment_title.append(post['name'])
                    
                    else:
                         
                        post_id.append(post['id'])
                        post_time.append(post['created_time'])
                        post_type.append(post['type'])
                        post_message.append('No Message for this Post')
                        post_attachment_title.append('No Title for this Attachment')

                    if 'description' in post:
                        
                        post_attachment_description.append(post['description'])

                    else:

                        post_attachment_description.append('No Description for this Attachment')
            
            else:
                
                counter=0
                global access_token
                
                # Send an E-Mail >>> See Instruction 2.1.
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.ehlo()
                server.starttls()
                server.login("SenderEMAIL@EMAIL.EMAIL", "PASSWORD")
                msg = "Hi,\n\n Page-Posts >>> Please paste a new Access Token!\n\n||Vahid S. J.||"
                server.sendmail("SenderEMAIL@EMAIL.EMAIL", "RecipientEMAIL@EMAIL.EMAIL", msg)
                server.quit()
                access_token = str(input("Please Enter a new SHORT-LIVED ACCESS TOKEN: "))
                graph = facebook.GraphAPI(access_token, version='2.3')
                posts = graph.get_connections(str(user), 'posts')
                
                if 'link' in post:
            
                    post_attachment_link.append(post['link'])
                    print(post['created_time'])
                    
                    if 'message' in post and 'name' in post:
                        post_id.append(post['id'])
                        post_time.append(post['created_time']) # The created Time is according to GMT!
                        post_type.append(post['type'])
                        post_message.append(post['message'])
                        post_attachment_title.append(post['name'])

                    elif 'message' in post and 'name' not in post:
                        
                        post_id.append(post['id'])
                        post_time.append(post['created_time']) # The created Time is according to GMT!
                        post_type.append(post['type'])
                        post_message.append(post['message'])
                        post_attachment_title.append('No Title for this Attachment')

                    elif 'message' not in post and 'name' in post:

                        post_id.append(post['id'])
                        post_time.append(post['created_time']) # The created Time is according to GMT!
                        post_type.append(post['type'])
                        post_message.append('No Message for this Post')
                        post_attachment_title.append(post['name'])
                    
                    else:
                         
                        post_id.append(post['id'])
                        post_time.append(post['created_time'])
                        post_type.append(post['type'])
                        post_message.append('No Message for this Post')
                        post_attachment_title.append('No Title for this Attachment')

                    if 'description' in post:
                        
                        post_attachment_description.append(post['description'])

                    else:

                        post_attachment_description.append('No Description for this Attachment')

                else:
                    
                    post_attachment_link.append('No Link for this Post')
                    print(post['created_time'])
                    
                    # some posts have no post description
                    if 'message' in post and 'name' in post:
                            
                        post_id.append(post['id'])
                        post_time.append(post['created_time']) # The created Time is according to GMT!
                        post_type.append(post['type'])
                        post_message.append(post['message'])
                        post_attachment_title.append(post['name'])

                    elif 'message' in post and 'name' not in post:
                        
                        post_id.append(post['id'])
                        post_time.append(post['created_time']) # The created Time is according to GMT!
                        post_type.append(post['type'])
                        post_message.append(post['message'])
                        post_attachment_title.append('No Title for this Attachment')

                    elif 'message' not in post and 'name' in post:

                        post_id.append(post['id'])
                        post_time.append(post['created_time']) # The created Time is according to GMT!
                        post_type.append(post['type'])
                        post_message.append('No Message for this Post')
                        post_attachment_title.append(post['name'])
                    
                    else:
                         
                        post_id.append(post['id'])
                        post_time.append(post['created_time'])
                        post_type.append(post['type'])
                        post_message.append('No Message for this Post')
                        post_attachment_title.append('No Title for this Attachment')

                    if 'description' in post:
                        
                        post_attachment_description.append(post['description'])

                    else:

                        post_attachment_description.append('No Description for this Attachment')

        # Wrap this block in a while loop so we can keep paginating requests until finished.
        global graph
        graph = facebook.GraphAPI(access_token, version='2.3')
        posts = graph.get_connections(str(user), 'posts')
        
        # Perform some action on each post in the collection we receive from Facebook.
        x = True
        while x == True:
            for post in posts['data']:
                last_PostID = id_query(post=post)
                if last_PostID not in newest_PostID:
                    post_query(post=post)
                else:
                    x = False
                    break
            # Attempt to make a request to the next page of data, if it exists.
            try:
                posts = requests.get(posts['paging']['next']).json()
            except:
                break
        
        # Create a table
        POSTS = Table().with_columns('Post ID', post_id,
                                     'Post Type', post_type,
                                     'Post-Created Time' , post_time,
                                     'Post Message', post_message,
                                     'Post Attachment Titel', post_attachment_title,
                                     'Post Attachment Description', post_attachment_description,
                                     'Post Attachment Link', post_attachment_link)

        # Save as a CSV file
        POSTS.to_csv(str(user)+'-Posts.csv')

    all_done_FileNames.append(str(filename))
    with open('all_done_FileNames.pkl', 'wb') as f:
        pickle.dump(all_done_FileNames, f, pickle.HIGHEST_PROTOCOL)

# Send an E-Mail >>> See Instruction 2.2.
server = smtplib.SMTP('smtp.gmail.com', 587)
server.ehlo()
server.starttls()
server.login("SenderEMAIL@EMAIL.EMAIL", "PASSWORD")
msg = "Hi,\n\n Page-Posts >>> Done! for all pages!\n\n||Vahid S. J.||"
server.sendmail("SenderEMAIL@EMAIL.EMAIL", "RecipientEMAIL@EMAIL.EMAIL", msg)
server.quit()
