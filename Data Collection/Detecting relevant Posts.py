from datascience import *
import numpy as np
import smtplib
import os

keywords = ['flüchtling', 'duldung','dublin-verfahren', 'schengen', 'flucht', 'frontex', 'migration', 'willkommenskultur', 'obergrenze','genfer konvention', 'integration',
            'asyl', 'abschiebung', 'ausländer', 'migrant', 'wir schaffen das', 'islam', 'balkan-route', 'rückführung', 'zuwander', 'einwander', 'bamf', 'schengen', 'visa',
            'grenze', 'schlepper', 'türkei-deal', 'silvesternacht', 'sexuelle übergriffe', 'syrer', 'ausländisch']

post_counter = 0
counter = 0

# Paste the file address of CSV files, Rember the Slash and Back Slash
file_address = str('C:/Users/Vahid/Desktop/allFacebookPosts')

for filename in os.listdir(str(file_address)):

    counter+=1
    print(counter)
    PostPages = Table.read_table(str(file_address)+ "/" + str(filename), encoding="ISO-8859-1")

    PostIDs = PostPages.column('Post ID')
    PostTypes = PostPages.column('Post Type')
    PostCreatedTime = PostPages.column('Post-Created Time')
    PostMessages = PostPages.column('Post Message')
    PostAttachmentsTitle = PostPages.column('Post Attachment Titel')
    PostAttachmentsDescription = PostPages.column('Post Attachment Description')
    PostAttachmentsLink = PostPages.column('Post Attachment Link')

    post_counter += len(PostIDs)

    # Outputs
    postIDs_selected = []
    postTypes_selected = []
    postCreatedTime_selected = []
    post_message_selected = []
    post_attachment_title_selected = []
    post_attachment_description_selected = []
    post_attachment_link_selected = []
    
    
    KEYWORD = ''
    KEYWORDS = []


    j= -1
    index_selected = []

    zipped = zip(PostMessages, PostAttachmentsTitle, PostAttachmentsDescription)

    # Find the posts included the keywords
    for p_m, p_a_t, p_a_d in zipped:
        j += 1
        for keyword in keywords:
            if keyword in p_m.lower() or keyword in p_a_t.lower() or keyword in p_a_d.lower():
                print(j)
                if j not in index_selected:
                    index_selected.append(j)
                KEYWORD = KEYWORD+str(keyword.capitalize())+','
        if KEYWORD:
            KEYWORDS.append(KEYWORD[:-1])
            KEYWORD = ''

    for i_s in index_selected:
        postIDs_selected.append(PostIDs[i_s])

    for i_s in index_selected:
        postTypes_selected.append(PostTypes[i_s])

    for i_s in index_selected:
        postCreatedTime_selected.append(PostCreatedTime[i_s])

    for i_s in index_selected:
        post_message_selected.append(str(PostMessages[i_s].replace("\n"," ").replace("\r", "")))

    for i_s in index_selected:
        post_attachment_title_selected.append(str(PostAttachmentsTitle[i_s].replace("\n"," ").replace("\r", "")))

    for i_s in index_selected:
        post_attachment_description_selected.append(str(PostAttachmentsDescription[i_s].replace("\n"," ").replace("\r", "")))

    for i_s in index_selected:
        post_attachment_link_selected.append(PostAttachmentsLink[i_s])


    # Create a table           
    RELEVANT_POSTS = Table().with_columns('Relevant Post ID', postIDs_selected,
                                          'Relevant Post Type', postTypes_selected,
                                          'Relevant Post-Created Time', postCreatedTime_selected,
                                          'Keyword', KEYWORDS,
                                          'Relevant Post Message', post_message_selected,
                                          'Relevant Post Attachment Titel', post_attachment_title_selected,
                                          'Relevant Post Attachment Description', post_attachment_description_selected,
                                          'Relevant Post Attachment Link', post_attachment_link_selected)

    # Save as a CSV file
    RELEVANT_POSTS.to_csv(str(filename)[:-10]+'-RelevantPosts.csv')

print("All Collected Posts: ", post_counter)

# Send an E-Mail >>> See Instruction 1.
server = smtplib.SMTP('smtp.gmail.com', 587)
server.ehlo()
server.starttls()
server.login("SenderEMAIL@EMAIL.EMAIL", "PASSWORD")
msg = "Hi,\n\nDONE! >>> Relevant Posts" + "\n\n||Vahid S. J.||"
server.sendmail("SenderEMAIL@EMAIL.EMAIL", "RecipientEMAIL@EMAIL.EMAIL", msg)
server.quit()
