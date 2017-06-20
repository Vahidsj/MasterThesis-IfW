import facebook
import requests
import os
from datascience import *
import smtplib


# You should paste your access_token here. You need the long-lived access token >>> See Instruction 2.2.
access_token = 'YOUR ACCESS TOKEN'
graph = facebook.GraphAPI(access_token, version='2.7')

# Paste the file address of CSV files, Remeber the Slash and Back Slash
file_address = "C:/Users\Vahid\Desktop\..."

for filename in os.listdir(str(file_address)):
    
    j=0
    
    try:
        # Outputs
        post_id = []
        comments_id = []
        comments_message = []
        post_message = []
        post_attachment = []

        PostPages = Table.read_table(str(file_address)+ "/" + str(filename), encoding="ISO-8859-1")
        PostIDs = PostPages.column('Relevant Post ID')
        PostMessages = PostPages.column('Relevant Post Message')
        PostAttachments = PostPages.column('Relevant Post Attachment Titel')

        zipped_1 = zip(PostIDs, PostMessages, PostAttachments)

        def comment_query(comment):
            
            post_id.append(str(p_id))
            post_message.append(str(p_m))
            post_attachment.append(str(p_a))
            comments_id.append(str(comment['id']))
            comments_message.append(str(comment['message'].replace("\n", " ").replace("\r", " ").replace("\t", " ")))
            
            print(str(comment['id']))
                
            return (post_id, post_message, post_attachment, comments_id, comments_message)

        # Collect the Comments
        for p_id, p_m, p_a in zipped_1:
            
            print(len(PostIDs))
            j +=1
            print(j)
            
            comments = graph.get_connections(str(p_id), 'comments')
            
            if not comments['data']:
                
                # some posts have no comments
                post_id.append(str(p_id))
                post_message.append(str(p_m))
                post_attachment.append(str(p_a))
                comments_id.append('No Comment for Post')
                comments_message.append('No Comment Message')
                print('No Comment', p_id)
            
            else:
                
                while True:
                    try:
                        # Perform some action on each post in the collection we receive from
                        # Facebook.
                        [comment_query(comment=comment) for comment in comments['data']]
                        # Attempt to make a request to the next page of data, if it exists.
                        comments = requests.get(comments['paging']['next']).json()
                    except KeyError:
                        print('No More Page for Comments!')
                        # When there are no more pages (['paging']['next']), break from the
                        # loop and end the script.
                        break

        zipped_2 = zip(post_id, comments_id)
        post_comment_id = []

        for p_i, c_i, in zipped_2:
            
            if c_i == 'No Comment for Post':
                
                post_comment_id.append('No Comment for Post')

            elif '_' not in c_i:

                post_comment_id.append(str(str(p_i)+'_'+str(c_i)))
                
            else:
                
                post_comment_id.append(str(str(p_i)+'_'+str(c_i.split('_')[1])))


        # Create a table
        post_comment = Table().with_columns('PostCommentID', post_comment_id,
                                            'Post Message', post_message,
                                            'Post Attachment', post_attachment,
                                            'Comment Message', comments_message)
        # Save as a CSV file
        post_comment.to_csv(str(filename)[:-19]+'-post-comments.csv')

            
    except Exception as e:
        
            # Send an E-Mail >>> See Instruction 2.1.
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.ehlo()
            server.starttls()
            server.login("SenderEMAIL@EMAIL.EMAIL", "PASSWORD")
            msg = "Hi,\n\nPost Comments >>> There is a problem for this page: "+ str(filename)[:-19] +"\and the problem is: "+str(e)+"\n\n||Vahid S. J.||"
            server.sendmail("SenderEMAIL@EMAIL.EMAIL", "RecipientEMAIL@EMAIL.EMAIL", msg)
            server.quit()

# Send an E-Mail >>> See Instruction 2.2.
server = smtplib.SMTP('smtp.gmail.com', 587)
server.ehlo()
server.starttls()
server.login("vahid.vsh.68@gmail.com", "vahid9050931")
msg = "Hi,\n\nDONE! >>> Post Comments for all the Pages\n\n||Vahid S. J.||"
server.sendmail("SenderEMAIL@EMAIL.EMAIL", "RecipientEMAIL@EMAIL.EMAIL", msg)
server.quit()
