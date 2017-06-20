import datascience
import facebook
import requests
from datascience import *
import smtplib

# I need here the short lived access token , because long lived access token is V2.6 and I need V2.3 to work the post_query
access_token = 'EAACEdEose0cBANDgsCy0mrTPWTpbWNVmLVsDRMOMxw5XZCqIP3RwlXE78mWVaHpneOMxugQyUTeUFi49UzxHXQRSM1rzUQUn7ZAsRxeEuKsOKIdtU507aCCDYc0sjxHmAY8sh4iG4DGtX6e0zcwwlegLOObKjR5PulbgnO5wZDZD'

# Put here the name of the facebook Page (e.g. https://www.facebook.com/bild.hamburg/ >>> bild.hamburg)

users = ['artede','heuteshow',
         'spiegelonline','extra3','linkspartei','linksfraktion','Change.orgDeutschland','Bundesregierung','meinRTL','AngelaMerkel','focus.de','deutschlandfunk',
         'dkultur','kmii.aktion','bamf.socialmedia','dw.deutschewelle','NDR.de','zeitonline','BuzzFeedDeutschland','taz.kommune','stern','HumanRightsWatchDeutschland',
         'AuswaertigesAmt','proasyl','diezeit','zeitonline','zeitcampus','Tageblatt.lu','tageblatt','szmagazin','ihre.sz','DerSpiegel','SPIEGEL.TV','managermagazin','BamS',
         'chrismon.evangelisch','SpiegelVideo','einestages','rponline','sat1nrw','JuedischeAllgemeine','NRW','waz','nordbayern.de','retter.tv','ZDF']
         


counter = 0

for user in users:
    
    # outputs
    post_id = []
    post_time = []
    post_type = []
    post_message = []
    post_attachment_title = []
    post_attachment_description = []
    post_attachment_link = []
    
    # Collecting the post id, post created time, post type, and post message (post description)    
    def post_query(post):
        
        global counter
        counter = counter+1
        
        if counter < 45000:
            
            print(counter)
                
            if 'link' in post:
                post_attachment_link.append(post['link'])
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
        
        else:
            
            counter=0
            global access_token
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.ehlo()
            server.starttls()
            server.login("vahid.vsh.68@gmail.com", "vahid9050931")
            msg = "Hi,\n\n Page-Posts >>> Access Token for this page:: "+str(user)+"\n\n||Vahid S. J.||"
            server.sendmail("vahid.vsh.68@gmail.com", "sadirijavadi.vahid@gmail.com", msg)
            server.quit()
            access_token = str(input("Please Enter a new SHORT-LIVED ACCESS TOKEN: "))
            print(access_token)
            graph = facebook.GraphAPI(access_token, version='2.3')
            posts = graph.get_connections(str(user), 'posts')
            print(access_token)
            
            if 'link' in post:
        
                post_attachment_link.append(post['link'])
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
    while True:
        try:
            # Perform some action on each post in the collection we receive from Facebook.
            [post_query(post=post) for post in posts['data']]
            # Attempt to make a request to the next page of data, if it exists.
            posts = requests.get(posts['paging']['next']).json()
        except KeyError:
            print('No More Post!')
            # When there are no more pages (['paging']['next']), break from the
            # loop and end the script.
            break

    POSTS = Table().with_columns('Post ID', post_id,
                                'Post Type', post_type,
                                'Post-Created Time' , post_time,
                                'Post Message', post_message,
                                'Post Attachment Titel', post_attachment_title,
                                'Post Attachment Description', post_attachment_description,
                                'Post Attachment Link', post_attachment_link)

    POSTS.to_csv(str(user)+'-Posts.csv')

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login("vahid.vsh.68@gmail.com", "vahid9050931")
    msg = "Hi,\n\n Page-Posts >>> Done! for this page:: "+str(user)+"\n\n||Vahid S. J.||"
    server.sendmail("vahid.vsh.68@gmail.com", "sadirijavadi.vahid@gmail.com", msg)
    server.quit()

server = smtplib.SMTP('smtp.gmail.com', 587)
server.ehlo()
server.starttls()
server.login("vahid.vsh.68@gmail.com", "vahid9050931")
msg = "Hi,\n\n Page-Posts >>> Done! for all pages:\n\n||Vahid S. J.||"
server.sendmail("vahid.vsh.68@gmail.com", "sadirijavadi.vahid@gmail.com", msg)
