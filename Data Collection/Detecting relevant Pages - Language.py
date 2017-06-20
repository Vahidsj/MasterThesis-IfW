import facebook
import requests
from datascience import *
from langdetect import detect
import numpy as np
import smtplib
import time
import os


# You should paste your access_token here. You need the short lived access token , because we don't have version 2.3 of the long lived access token. post_query function works faster with this version. >>> See Instruction 1.
access_token = 'YOUR ACCESS TOKEN'
graph = facebook.GraphAPI(access_token, version='2.6')

irrelevant_categories = ['Coach', 'Business Service', 'Sports Team', 'Consulting/Business Service', 'Electronics', 'Local Business', 'Real Estate', 'Internet/Software',
                         'Software', 'Computers/Technology', 'Games/Toys', 'Shopping/Retail', 'Home Decor', 'Home/Garden Website', 'Museum/Art Gallery', 'Wine/Spirits',
                         'Company', 'Professional Service', 'Health/Beauty', 'Clothing', 'Just For Fun', 'Photographer', 'Movie Theater', 'Engineering/Construction',
                         'Bar','Bank/Financial Institution','EnterTrainer Coach','Travel/Leisure','Business/Economy Website','Appliances', 'Restaurant/Cafe', 'Food/Beverages',
                         'Store', 'Jewelry/Watches', 'Transportation', 'Landmark', 'Hotel', 'Retail and Consumer Merchandise', 'Automotive', 'Automobiles and Parts',
                         'Sports & Recreation', 'Sports League', 'Sports Venue & Stadium', 'Spas/Beauty/Personal Care', 'Health/Beauty', 'Institution', 'Bank/Financial Service',
                         'Health/Medical/Pharmaceuticals', 'Lawyer', 'Legal/Law', 'Business Person', 'Attractions/Things to Do', 'Small Business', 'Tours & Sightseeing',
                         'Performance Venue', 'Arts & Entertainment', 'Personal Blog', 'Book Store', 'Grocery Store', 'Baby Goods/Kids Goods', 'Cause', 'Public Places', 'Book',
                         'Supplies', 'Movie', 'Design', 'Event Planner', 'Concert Tour', 'Club', 'Record Label', 'Music Award', 'Cars', 'Landmark', 'Arts/Humanities Website',
                         'Aerospace/Defense', 'Library', 'Outdoor Gear/Sporting Goods', 'Monarch', 'Fictional Character', 'Cargo & Freight Company', 'Chef', 'Drink', 'Video Game',
                         'Phone/Tablet', 'App Page', 'Energy', 'Musician/Band', 'Dancer', 'Playlist', 'Bar', 'Household Supplies', 'Product/Service', 'Telecommunication', 'Local/Travel Website',
                         'Visual Arts','Computers/Internet Website', 'Recreation/Sports Website', 'Performance Art', 'Kitchen/Cooking', 'Language', 'Publisher', 'Furniture', 'TV/Movie Award',
                         'Movie & Television Studio', 'Camera/Photo', 'Farming/Agriculture', 'Reference Website', 'Tools/Equipment', 'Health/Wellness Website', 'Sports Event', 'Airport',
                         'Transit Stop', 'Bags/Luggage', 'Music Video', 'Pet Supplies', 'Insurance Company', 'Pet Service', 'Art', 'Science Website', 'Education Website', 'Other', 'Lake',
                         'Amateur Sports Team', 'Field of Study', 'Country', 'Teens/Kids Website', 'State/Province/Region', 'Event Planning Service', 'Album', 'Medical & Health', 'Doctor',
                         'Musical Genre', 'Holiday','City', 'Profession', 'Designer', 'Music', 'Scientist', 'Sport', 'School Sports Team', 'Vitamins/Supplements', 'Teacher',
                         'Drugs', 'Music Chart', 'Cuisine', 'Mountain', 'Movie Genre', 'Pet', 'Work Position', 'Professional Sports Team', 'Computers', 'Food', 'Food & Beverage Company',
                         'Movie Character', 'Literary Editor', 'Clothing (Brand)', 'Island', 'Diseases', 'E-commerce Website', 'Recreation & Fitness', 'Home Improvement',
                         'Retail Company', 'Work Status', 'Song', 'Industrials', 'Neighborhood', 'Patio/Garden', 'Building Materials', 'Book Series', 'Hospital/Clinic',
                         'Mining/Materials', 'Internet Company', 'Hotel & Lodging', 'Animal', 'Musical Instrument', 'Book Genre', 'Biotechnology', 'Advertising/Marketing Service',
                         'Commercial Equipment', 'Automotive Company', 'Continent', 'Animal Breed', 'Episode', 'Chemicals', 'Beauty', 'Board Game', 'Marine', 'Geographical feature',
                         'Office Supplies', 'Consulting Agency', 'Brand/Company Type', 'Church/Religious Organization', 'Competition', 'Entrepreneur', 'Film Director', 'High School Status',
                         'Industrial Company', 'Local Service', 'Medical Company', 'Producer', 'Travel Company', 'Train Station', 'TV Genre', 'Elementary School', 'Video']



for filename in os.listdir('C:/Users/Vahid/Desktop/...'):
    
    LikePages = Table.read_table("C:/Users/Vahid/Desktop/Relevant Pages-unmodified/"+str(filename), encoding="ISO-8859-1")

    IDS = LikePages.column(0)
    page_liked_name = LikePages.column(1)
    page_liked_category = LikePages.column(2)
    page_liked_about = LikePages.column(3)
    page_liked_description = LikePages.column(4)
    page_liked_products = LikePages.column(5)
    page_liked_link = LikePages.column(6)

    relevant_page_ids_modified = []
    page_liked_name_selected = []
    page_liked_category_selected = []
    page_liked_about_selected = []
    page_liked_description_selected = []
    page_liked_products_selected = []
    page_liked_link_selected = []
    Problematic_Page_IDs = []


    ID = []

    for i in IDS:
        ID.append(i[:-1])

    post_message = []
    pageIDs_selected = []
    j=0

    def post_query(post):
        
        if 'message' in post:
            post_message.append(post['message'])


    for p_id in ID:

        print(len(ID))
        print(j)
        j+=1
        try:
            page_profile = graph.get_object(str(p_id), fields='category')
            print(p_id)

            if page_profile['category'] not in irrelevant_categories:

                posts = graph.get_connections(str(p_id), 'posts')
                [post_query(post=post) for post in posts['data']]
                
                if len(post_message) >= 5:
                    
                    try:
                        post_language1 = detect(str(post_message[0]))
                        post_language2 = detect(str(post_message[2]))
                        post_language3 = detect(str(post_message[4]))
                        
                        if post_language1 == 'de' or post_language2 == 'de' or post_language3 == 'de':
                            relevant_page_ids_modified.append(str(p_id)+'_')
                            index_selected = ID.index(str(p_id))
                            page_liked_name_selected.append(page_liked_name[index_selected])
                            page_liked_category_selected.append(page_liked_category[index_selected])
                            page_liked_about_selected.append(page_liked_about[index_selected])
                            page_liked_description_selected.append(page_liked_description[index_selected])
                            page_liked_products_selected.append(page_liked_products[index_selected])
                            page_liked_link_selected.append(page_liked_link[index_selected])
                    except:
                            relevant_page_ids_modified.append(str(p_id)+'_')
                            index_selected = ID.index(str(p_id))
                            page_liked_name_selected.append(page_liked_name[index_selected])
                            page_liked_category_selected.append(page_liked_category[index_selected])
                            page_liked_about_selected.append(page_liked_about[index_selected])
                            page_liked_description_selected.append(page_liked_description[index_selected])
                            page_liked_products_selected.append(page_liked_products[index_selected])
                            page_liked_link_selected.append(page_liked_link[index_selected])
                
                elif len(post_message) <= 3 and len(post_message) >= 1:
                    
                    try:
                        
                        post_language1 = detect(str(post_message[0]))
                        post_language2 = detect(str(post_message[2]))
                  
                        if post_language1 == 'de' or post_language2 == 'de':
                            relevant_page_ids_modified.append(str(p_id)+'_')
                            index_selected = ID.index(str(p_id))
                            page_liked_name_selected.append(page_liked_name[index_selected])
                            page_liked_category_selected.append(page_liked_category[index_selected])
                            page_liked_about_selected.append(page_liked_about[index_selected])
                            page_liked_description_selected.append(page_liked_description[index_selected])
                            page_liked_products_selected.append(page_liked_products[index_selected])
                            page_liked_link_selected.append(page_liked_link[index_selected])
                    except:
                        relevant_page_ids_modified.append(str(p_id)+'_')
                        index_selected = ID.index(str(p_id))
                        page_liked_name_selected.append(page_liked_name[index_selected])
                        page_liked_category_selected.append(page_liked_category[index_selected])
                        page_liked_about_selected.append(page_liked_about[index_selected])
                        page_liked_description_selected.append(page_liked_description[index_selected])
                        page_liked_products_selected.append(page_liked_products[index_selected])
                        page_liked_link_selected.append(page_liked_link[index_selected])
                else:
                  
                    relevant_page_ids_modified.append(str(p_id)+'_')
                    index_selected = ID.index(str(p_id))
                    page_liked_name_selected.append(page_liked_name[index_selected])
                    page_liked_category_selected.append(page_liked_category[index_selected])
                    page_liked_about_selected.append(page_liked_about[index_selected])
                    page_liked_description_selected.append(page_liked_description[index_selected])
                    page_liked_products_selected.append(page_liked_products[index_selected])
                    page_liked_link_selected.append(page_liked_link[index_selected])
                    
                post_message = []

        except Exception as e:
            
            Problematic_Page_IDs.append(str(p_id))
            # Send an E-Mail >>> See Instruction 3.1.
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.ehlo()
            server.starttls()
            server.login("SenderEMAIL@EMAIL.EMAIL", "PASSWORD")
            msg = "Hi,\n\n Page-Posts >>> Please paste a new Access Token!\n\n||Vahid S. J.||"
            server.sendmail("SenderEMAIL@EMAIL.EMAIL", "RecipientEMAIL@EMAIL.EMAIL", msg)
            server.quit()
            time.sleep(120)


               
    LIKED_PAGES = Table().with_columns('Relevant Page', relevant_page_ids_modified,
                                       'page_liked_name', page_liked_name_selected,
                                       'page_liked_category', page_liked_category_selected,
                                       'page_liked_about', page_liked_about_selected,
                                       'page_liked_description', page_liked_description_selected,
                                       'page_liked_products', page_liked_products_selected,
                                       'page_liked_link', page_liked_link_selected)

    LIKED_PAGES.to_csv(str(filename))
     
    # Send an E-Mail >>> See Instruction 3.1.
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login("SenderEMAIL@EMAIL.EMAIL", "PASSWORD")
     
    msg = "Hi,\n\nDetecting relevant Pages --->>> DONE!\nFor this Page: "+str(filename)+"\n\n||Vahid S. J.||"

    server.sendmail("SenderEMAIL@EMAIL.EMAIL", "RecipientEMAIL@EMAIL.EMAIL", msg)
    server.quit()
