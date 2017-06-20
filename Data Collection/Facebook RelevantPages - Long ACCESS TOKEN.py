import facebook
import requests
from datascience import *
from langdetect import detect
import numpy as np
import smtplib
import time
global graph
        
# We need here the long lived access token
access_token = 'EAAI6JhLteM8BAHiIE4YZCv5OfAjugOw9W2xVnZBbxIZCA4KEZB4joRTDLT7vs3GteDs8F4nbM3wY09hDNQRhJZBUVltc3qrUgA2E8ULnwiaGcR1T7cKp7Yr403TZAmLXtbnnRKBB0XRPS5aFYlxB8iVak9ZBZBNKZA30ZD'
graph = facebook.GraphAPI(access_token, version='2.6')

# Put here the name of the facebook Page (e.g. https://www.facebook.com/bild.hamburg/ >>> bild.hamburg)
# users = ['wzwuppertal', 'WAZBochum', 'HersbruckerZeitung', 'westfaelischer.anzeiger', 'LNOnline', 'reportMuenchen', 'abendblatt', 'badischezeitung.de', 'berlinerzeitung', 'bild',
#'FrankfurterNeuePresse' ,'gaonline', 'kielernachrichten', 'ksta.fb', 'lvzonline', 'mannheimer.morgen', 'morgenpost', 'RemscheiderGA', 'WAZEssen', 'wnonline',
#'mzwebde', 'siegenerzeitung', 'Erlanger.Nachrichten.Online', 'tlz.de', 'Volksfreund', 'nordseezeitung', 'RZKoblenz', 'gea.reutlingen','goettingertageblatt', 'pznews',
#'junge.welt', 'rp.duesseldorf', 'stuttgarternachrichten', 'duesseldorf.feuerwehr', 'wznewsline', 'tag24.bielefeld', 'tag24.minden', 'tag24.paderborn',
#'tag24.chemnitz', 'tag24.leipzig', 'tag24.zwickau', 'tag24.erzgebirge', 'tag24.vogtland', 'tag24.erfurt', 'tag24.dresden', 'szonline',
#'HannoverscheAllgemeine', 'Nuernberger.Nachrichten.Online', 'rp.duisburg', 'tag24.gera', 'tag24.jena', 'WAZDuisburg', 'WESER.KURIER', '115821908508213', 'delmenhorster.kreisblatt',
#'BILD.Bremen', 'Bremen', 'aachenernachrichten', 'aachenerzeitung', 'rshradio', 'rp.leverkusen', 'MAZonline', 'PNN.de', 'nnnonline', 'oz.rostock',
#'taerfurt', 'nachrichten.in.kiel', 'shzonline', 'nuernberger.zeitung', 'huffingtonpostdepolitik', 'bergedorferzeitung', 'hamburgermorgenpost', 'mainpost',
#'HeilbronnerStimme', 'suedwestpresse', 'wolfsburgerallgemeine', 'donaukurier.online', 'SWitySW', 'mittelbayerische', 'westfalenblatt', 'echoonline',
#'RheinNeckarZeitung', 'delmenhorster.kreisblatt', 'ksta.lev', 'wuerzburg.erleben', 'blaulicht.wuerzburg', 'SchleiBote', 'schleswiger.nachrichten',
#'166635820118970', 'nordfriesland.tageblatt', '279156428824958', 'flensburger.tageblatt', 'husumernachrichten', 'SchleswigHolstein', 'courier.nms',
#'293833847330546', '1617631925184537', 'LNSegeberg', 'BremenVier', 'bild.hamburg', 'welthamburg', 'BILD.BB', 'huffingtonpostde', 'muenchner.merkur', 'SZmuenchen',
#'sonntagsblatt.bayern', 'tzmuenchen', 'muenchner.merkur.landkreis', 'blaulichtinberlin', 'imwestenberlins', 'zittyberlin', 'bkurier',
#'tagesspiegel', 'prenzlauerbergnachrichten', 'derfreitag', '482499455210420', 'reporterohnegrenzen', 'tip.Berlin', 'berichtausberlin', 'rp.koeln',
#'schongauer.nachrichten', 'erdinger.dorfener.anzeiger', 'freisinger.tagblatt', 'dachauer.nachrichten', 'fuerstenfeldbrucker.tagblatt', 'gap.tagblatt',
#'rundschau.online', 'faz', 'stuttgarterzeitung', 'dnnonline', 'NNLokales', 'pegnitzzeitung', 'kreiszeitung.de', 'DiepholzerKreisblatt',
#'wochenwebschau', 'Weserreport', 'Weserreport', 'VerdenerAllerZeitung', 'rotenburger.kreiszeitung','11freunde','jungundnaiv','welt','netzpolitik','DasErste','ARD',
#'tagesschau','Tatort','funkhauseuropa','artede','heuteshow','PULS','linksfraktion','Change.orgDeutschland','Bundesregierung','meinRTL',
#'AngelaMerkel','focus.de','deutschlandfunk','dkultur','kmii.aktion','bamf.socialmedia','dw.deutschewelle','NDR.de','zeitonline','BuzzFeedDeutschland','taz.kommune',
#'stern','HumanRightsWatchDeutschland','AuswaertigesAmt','proasyl','diezeit','zeitonline','zeitcampus','Tageblatt.lu','tageblatt','szmagazin','ihre.sz','DerSpiegel',
#'SPIEGEL.TV','managermagazin','BamS','chrismon.evangelisch','SpiegelVideo','einestages','rponline','sat1nrw','JuedischeAllgemeine','NRW','waz','nordbayern.de',
#'retter.tv','ZDF', 'toelzer.kurier','bayern3'']


users = ['linkspartei']

for user in users:

    ALL_PAGE_IDS = Table.read_table('AllLikePages-Explored.csv', encoding="ISO-8859-1").column(0)
    all_page_ids = []

    #All already explored pages
    for API in ALL_PAGE_IDS:
        all_page_ids.append(API[:-1])

    #All already explored pages in this level, Exploring 1
    EXPLORING_IDS_1 = Table.read_table('Exploring_IDs_1.csv', encoding="ISO-8859-1").column(0)
    exploring_ids_1 = []
    for EI1 in EXPLORING_IDS_1:
        exploring_ids_1.append(EI1[:-1])

    #All already explored pages in this level, Exploring 2
    EXPLORING_IDS_2 = Table.read_table('Exploring_IDs_2.csv', encoding="ISO-8859-1").column(0)
    exploring_ids_2 = []
    for EI2 in EXPLORING_IDS_2:
        exploring_ids_2.append(EI2[:-1])

    #Outputs
    relevant_page_ids = []
    exploring_ids_1_new = []
    exploring_ids_2_new = []
    page_liked_name = []
    page_liked_about = []
    page_liked_fan_count = []
    page_liked_category = []
    page_liked_description = []
    page_liked_products = []
    page_liked_link = []

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

    profile = graph.get_object(user, fields= 'id, name, fan_count, category, link, about, description, products')

    relevant_page_ids.append(str(profile['id']))
    all_page_ids.append(str(profile['id']))
    page_liked_name.append(profile['name'])
    page_liked_fan_count.append(profile['fan_count'])
    page_liked_category.append(profile['category'])
    page_liked_link.append(profile['link'])


    try:
                                        
        page_liked_about.append(profile['about'])
                                        
    except KeyError:
                                        
        page_liked_about.append('No Page liked about')

    try:
                                        
        page_liked_description.append(profile['description'])
                                        
    except KeyError:
                                        
        page_liked_description.append('No Page liked Description')

    try:
                                        
        page_liked_products.append(profile['products'])
                                        
    except KeyError:
                                        
        page_liked_products.append('No Page liked Products')



    pages_liked = graph.get_connections(profile['id'], 'likes')

    # Collecting the page id, page category, page fan_count etc.
    def page_liked_query(page_liked):
        
        if page_liked['id'] not in exploring_ids_1:
            
            try:
                
                page_liked_profile = graph.get_object(str(page_liked['id']), fields= 'category')

                if page_liked_profile['category'] not in irrelevant_categories:
                    
                    exploring_ids_1.append(str(page_liked['id']))
                    exploring_ids_1_new.append(str(page_liked['id']))
            
            except Exception as e:
                
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.ehlo()
                server.starttls()
                server.login("vahid.vsh.68@gmail.com", "vahid9050931")
                msg = "Hi,\n\n Page-liked --->>> There is a problem for this page ID: "+str(page_liked['id'])+" and the problem is: "+str(e)+"\nPlease COME TO ME!\n\n||Vahid S. J.||"
                server.sendmail("vahid.vsh.68@gmail.com", "sadirijavadi.vahid@gmail.com", msg)
                server.quit()
            
        if page_liked['id'] not in all_page_ids:

            try:    
                all_page_ids.append(str(page_liked['id']))
                page_liked_profile = graph.get_object(str(page_liked['id']), fields= 'id, name, fan_count, category, link, about, description, products')        
                    
                    
                if page_liked_profile['fan_count'] >= 3000 and page_liked_profile['category'] not in irrelevant_categories:
                                       
                    print('Relevant', page_liked_profile['name'], "-->>", page_liked_profile['category'])
                    relevant_page_ids.append(page_liked_profile['id'])
                    page_liked_name.append(page_liked_profile['name'])
                    page_liked_fan_count.append(page_liked_profile['fan_count'])
                    page_liked_category.append(page_liked_profile['category'])
                    page_liked_link.append(page_liked_profile['link'])
                                                
                    try:
                                                  
                        page_liked_about.append(page_liked_profile['about'])
                                                    
                    except KeyError:
                                                    
                        page_liked_about.append('No Page liked about')

                    try:
                                                    
                        page_liked_description.append(page_liked_profile['description'])
                                                    
                    except KeyError:
                                                    
                        page_liked_description.append('No Page liked Description')
                                            
                    try:
                                                    
                        page_liked_products.append(page_liked_profile['products'])
                                                    
                    except KeyError:
                                                    
                        page_liked_products.append('No Page liked Products')
                                                
                else:

                    print('Irrelevant', page_liked_profile['name'], "-->> ", page_liked_profile['category'])

            except Exception as e:

                if e == "(#17) User request limit reached":
                    
                    global graph
                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.ehlo()
                    server.starttls()
                    server.login("vahid.vsh.68@gmail.com", "vahid9050931")
                    msg = "Hi,\n\n Page-liked --->>> There is a problem for this page ID: "+str(page_liked['id'])+" and the problem is: "+str(e)+"\nI'm changing the access token in order to work the script again!\n\n||Vahid S. J.||"
                    server.sendmail("vahid.vsh.68@gmail.com", "sadirijavadi.vahid@gmail.com", msg)
                    server.quit()
                    access_token = ""
                    graph = facebook.GraphAPI(access_token, version='2.8')
                    time.sleep(3650)

                else:
                    
                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.ehlo()
                    server.starttls()
                    server.login("vahid.vsh.68@gmail.com", "vahid9050931")
                    msg = "Hi,\n\n Page-liked --->>> There is a problem for this page ID: "+str(page_liked['id'])+" and the problem is: "+str(e)+"\nPlease COME TO ME!\n\n||Vahid S. J.||"
                    server.sendmail("vahid.vsh.68@gmail.com", "sadirijavadi.vahid@gmail.com", msg)
                    server.quit()
        else:
            print("This page: "+str(page_liked['id'])+" has been already explored whether it is relevant or not!")
        

        return(all_page_ids, relevant_page_ids, page_liked_name, page_liked_category, page_liked_about, page_liked_description, page_liked_products, page_liked_link, exploring_ids_1, exploring_ids_1_new)



    ##### some posts have no page liked
    if not pages_liked['data']:
                
        print('No Page liked')
                
    else:
                
        ##### Wrap this block in a while loop so we can keep paginating requests until
        ##### finished.
        while True:
                        
            try:

                # Perform some action on each post in the collection we receive from
                # Facebook.
                [page_liked_query(page_liked=page_liked) for page_liked in pages_liked['data']]
                # Attempt to make a request to the next page of data, if it exists.
                pages_liked = requests.get(pages_liked['paging']['next']).json()

            except KeyError:
                print('No More Page!')
                # When there are no more pages (['paging']['next']), break from the
                # loop and end the script.
                break



    def page_liked_query_1(page_liked_1):
            
        if str(page_liked_1['id']) not in exploring_ids_1 and str(page_liked_1['id']) not in exploring_ids_2:
                
            try:
                
                page_liked_profile_1 = graph.get_object(str(page_liked_1['id']), fields='category')

                if page_liked_profile_1['category'] not in irrelevant_categories:
                
                    exploring_ids_2.append(str(page_liked_1['id']))
                    exploring_ids_2_new.append(str(page_liked_1['id']))

            except Exception as e:
            
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.ehlo()
                server.starttls()
                server.login("vahid.vsh.68@gmail.com", "vahid9050931")
                msg = "Hi,\n\n Page-liked --->>> There is a problem for this page ID: "+str(page_liked_1['id'])+" and the problem is: "+str(e)+"\nPlease COME TO ME!\n\n||Vahid S. J.||"
                server.sendmail("vahid.vsh.68@gmail.com", "sadirijavadi.vahid@gmail.com", msg)
                server.quit()

        if page_liked_1['id'] not in all_page_ids:

            try:
                
                page_liked_profile_1 = graph.get_object(str(page_liked_1['id']), fields= 'id, name, fan_count, category, link, about, description, products')      
                all_page_ids.append(str(page_liked_1['id']))
                
                if page_liked_profile_1['fan_count'] >= 3000 and page_liked_profile_1['category'] not in irrelevant_categories: ## Here can be modified
                                                
                    print('Relevant', page_liked_profile_1['name'], "-->>", page_liked_profile_1['category'])
                    relevant_page_ids.append(page_liked_profile_1['id'])
                    page_liked_name.append(page_liked_profile_1['name'])
                    page_liked_fan_count.append(page_liked_profile_1['fan_count'])
                    page_liked_category.append(page_liked_profile_1['category'])
                    page_liked_link.append(page_liked_profile_1['link'])
                                                
                    try:
                                            
                        page_liked_about.append(page_liked_profile_1['about'])
                                            
                    except KeyError:
                                            
                        page_liked_about.append('No Page liked about')

                    try:
                                            
                        page_liked_description.append(page_liked_profile_1['description'])
                                            
                    except KeyError:
                                            
                        page_liked_description.append('No Page liked Description')
                                        
                    try:
                                            
                        page_liked_products.append(page_liked_profile_1['products'])
                                        
                    except KeyError:
                                            
                        page_liked_products.append('No Page liked Products')
                                                
                else:
                                            
                    print('Irrelevant', page_liked_profile_1['name'], "-->> ", page_liked_profile_1['category'])

            except Exception as e:
                
                if e == "(#17) User request limit reached":
                    global graph
                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.ehlo()
                    server.starttls()
                    server.login("vahid.vsh.68@gmail.com", "vahid9050931")
                    msg = "Hi,\n\n Page-liked --->>> There is a problem for this page ID: "+str(page_liked_1['id'])+" and the problem is: "+str(e)+"\nI'm changing the access token in order to work the script again!\n\n||Vahid S. J.||"
                    server.sendmail("vahid.vsh.68@gmail.com", "sadirijavadi.vahid@gmail.com", msg)
                    server.quit()
                    access_token = "EAAPsgFwopnEBAI19K0TlAbWOnN2VCP4gjklg6v82tV2uxajwakBgpCb8HAui1gZBGQcq5tJIFyL1XmiUYQtUOffqKOJXyvZAYa1AcZBQgZBlGi2kR9Hz6qX8oay07JbOfaR9UZBLFaiAEXWwkZB9EsN6ZBZAsSFsxn0ZD"
                    graph = facebook.GraphAPI(access_token, version='2.8')
                    time.sleep(3650)

                else:

                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.ehlo()
                    server.starttls()
                    server.login("vahid.vsh.68@gmail.com", "vahid9050931")
                    msg = "Hi,\n\n Page-liked --->>> There is a problem for this page ID: "+str(page_liked_1['id'])+" and the problem is: "+str(e)+"\nPlease COME TO ME!\n\n||Vahid S. J.||"
                    server.sendmail("vahid.vsh.68@gmail.com", "sadirijavadi.vahid@gmail.com", msg)
                    server.quit()
        else:
            print("This page: "+str(page_liked_1['id'])+" has been already explored whether it is relevant or not!")
            
        return(all_page_ids, relevant_page_ids, page_liked_name, page_liked_category, page_liked_about, page_liked_description, page_liked_products, page_liked_link, exploring_ids_2, exploring_ids_2_new)

    # Check the page whether it is a relevant page or irrelevant

    post_message = []
    exploring_ids_1_selected = []

    def post_query(post):
                
        if 'message' in post:
            post_message.append(post['message'])

    for p_id in exploring_ids_1_new:
        posts = graph.get_connections(str(p_id), 'posts')
        [post_query(post=post) for post in posts['data']]
        print(p_id)
                
        if len(post_message) >= 5:
            
            try:
                post_language1 = detect(str(post_message[0]))
                post_language2 = detect(str(post_message[2]))
                post_language3 = detect(str(post_message[4]))
                if post_language1 == 'de' or post_language2 == 'de' or post_language3 == 'de':
                    exploring_ids_1_selected.append(str(p_id))
            except:
                exploring_ids_1_selected.append(str(p_id))
                
        elif len(post_message) <= 3 and len(post_message) >= 1:
                    
            try:
                post_language1 = detect(str(post_message[0]))
                post_language2 = detect(str(post_message[2]))
                if post_language1 == 'de' or post_language2 == 'de':
                    exploring_ids_1_selected.append(str(p_id))
            except:
                exploring_ids_1_selected.append(str(p_id))
        else:
            exploring_ids_1_selected.append(str(p_id))
                    
        post_message = [] 

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login("vahid.vsh.68@gmail.com", "vahid9050931")
    msg = "Hi,\n\n Page-liked --->>> Script is working so far very well and the number of pages for exploring-1: "+str(len(exploring_ids_1_selected)+"\n\n||Vahid S. J.||")
    server.sendmail("vahid.vsh.68@gmail.com", "sadirijavadi.vahid@gmail.com", msg)
    server.quit()            
    # Explore the selected pages
    print(len(exploring_ids_1_new)) #For Testing
    for p_id in exploring_ids_1_selected:
        print(len(exploring_ids_1_selected)) #For Testing
        pages_liked_1 = graph.get_connections(str(p_id), 'likes')
        i+=1 #For Testing
        print(i)            
        if not pages_liked_1['data']:
                        
            print('No Page liked')
                
        else:
                        
            ##### Wrap this block in a while loop so we can keep paginating requests until
            ##### finished.
                            
            while True:

                    try:

                        # Perform some action on each post in the collection we receive from
                        # Facebook.
                        [page_liked_query_1(page_liked_1=page_liked) for page_liked in pages_liked_1['data']]
                        # Attempt to make a request to the next page of data, if it exists.
                        pages_liked_1 = requests.get(pages_liked_1['paging']['next']).json()

                    except KeyError:
                        print('No More Page!')
                        # When there are no more pages (['paging']['next']), break from the
                        # loop and end the script.
                        break

    def page_liked_query_2(page_liked_2):
                    
        if str(page_liked_2['id']) not in all_page_ids:

            try:
                
                all_page_ids.append(str(page_liked_2['id']))
                page_liked_profile_2 = graph.get_object(str(page_liked_2['id']), fields= 'id, name, fan_count, category, link, about, description, products')
                
                if page_liked_profile_2['fan_count'] >= 3000 and page_liked_profile_2['category'] not in irrelevant_categories: ## Here can be modified
                                                
                    print('Relevant', page_liked_profile_2['name'], page_liked_profile_2['category'])
                    relevant_page_ids.append(page_liked_profile_2['id'])
                    page_liked_name.append(page_liked_profile_2['name'])
                    page_liked_fan_count.append(page_liked_profile_2['fan_count'])
                    page_liked_category.append(page_liked_profile_2['category'])
                    page_liked_link.append(page_liked_profile_2['link'])
                                                
                    try:
                                            
                        page_liked_about.append(page_liked_profile_2['about'])
                                            
                    except KeyError:
                                            
                        page_liked_about.append('No Page liked about')

                    try:
                                            
                        page_liked_description.append(page_liked_profile_2['description'])
                                            
                    except KeyError:
                                            
                        page_liked_description.append('No Page liked Description')
                                        
                    try:
                                            
                        page_liked_products.append(page_liked_profile_2['products'])
                                
                    except KeyError:

                        page_liked_products.append('No Page liked Products')
                                                
                else:
                                                
                    print('Irrelevant', page_liked_profile_2['name'], "-->> ", page_liked_profile_2['category'])

            except Exception as e:

                if e == "(#17) User request limit reached":
                    global graph
                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.ehlo()
                    server.starttls()
                    server.login("vahid.vsh.68@gmail.com", "vahid9050931")
                    msg = "Hi,\n\n Page-liked --->>> There is a problem for this page ID: "+str(page_liked_2['id'])+" and the problem is: "+str(e)+"\nI'm changing the access token in order to work the script again!\n\n||Vahid S. J.||"
                    server.sendmail("vahid.vsh.68@gmail.com", "sadirijavadi.vahid@gmail.com", msg)
                    server.quit()
                    access_token = "EAAPsgFwopnEBAI19K0TlAbWOnN2VCP4gjklg6v82tV2uxajwakBgpCb8HAui1gZBGQcq5tJIFyL1XmiUYQtUOffqKOJXyvZAYa1AcZBQgZBlGi2kR9Hz6qX8oay07JbOfaR9UZBLFaiAEXWwkZB9EsN6ZBZAsSFsxn0ZD"
                    graph = facebook.GraphAPI(access_token, version='2.8')
                    time.sleep(3650)

                else:
                    
                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.ehlo()
                    server.starttls()
                    server.login("vahid.vsh.68@gmail.com", "vahid9050931")
                    msg = "Hi,\n\n Page-liked --->>> There is a problem for this page ID: "+str(page_liked_2['id'])+" and the problem is: "+str(e)+"\nPlease COME TO ME!\n\n||Vahid S. J.||"
                    server.sendmail("vahid.vsh.68@gmail.com", "sadirijavadi.vahid@gmail.com", msg)
                    server.quit()       
        else:
            print("This page: "+str(page_liked_2['id'])+" has been already explored whether it is relevant or not!")
        
        return(all_page_ids, relevant_page_ids, page_liked_name, page_liked_category, page_liked_about, page_liked_description, page_liked_products, page_liked_link)


    post_message = []
    exploring_ids_2_selected = []

    def post_query(post):
                
        if 'message' in post:
            post_message.append(post['message'])

    for p_id in exploring_ids_2_new:
        
        posts = graph.get_connections(str(p_id), 'posts')
        [post_query(post=post) for post in posts['data']]
        print(p_id)
                
        if len(post_message) >= 5:
            try:
                post_language1 = detect(str(post_message[0]))
                post_language2 = detect(str(post_message[2]))
                post_language3 = detect(str(post_message[4]))
                if post_language1 == 'de' or post_language2 == 'de' or post_language3 == 'de':
                    exploring_ids_2_selected.append(str(p_id))
            except:
                exploring_ids_2_selected.append(str(p_id))
                
        elif len(post_message) <= 3 and len(post_message) >= 1:
                    
            try:
                post_language1 = detect(str(post_message[0]))
                post_language2 = detect(str(post_message[2]))
                if post_language1 == 'de' or post_language2 == 'de':
                    exploring_ids_2_selected.append(str(p_id))
            except:
                exploring_ids_2_selected.append(str(p_id))
        else:
            exploring_ids_2_selected.append(str(p_id))
                    
        post_message = []
            
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login("vahid.vsh.68@gmail.com", "vahid9050931")
    msg = "Hi,\n\n Page-liked --->>> Script is working so far very well and the number of pages for exploring-2: "+str(len(exploring_ids_2_selected)+"\n\n||Vahid S. J.||")
    server.sendmail("vahid.vsh.68@gmail.com", "sadirijavadi.vahid@gmail.com", msg)
    server.quit()
    
    # Explore the selected pages
    print(len(exploring_ids_2_new)) #For Testing
    for p_id in exploring_ids_2_selected:
        print(len(exploring_ids_2_selected)) #For Testing
        pages_liked_2 = graph.get_connections(str(p_id), 'likes')
        j+=1 #For Testing
        print(j)

        if j == 1000 or j==2000 or j==3000 or j==4000:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.ehlo()
            server.starttls()
            server.login("vahid.vsh.68@gmail.com", "vahid9050931")
            msg = "Hi,\n\n Page-liked --->>> Script is working very well! and "+str(j)+" Pages have been explored up to now!\n\n||Vahid S. J.||"
            server.sendmail("vahid.vsh.68@gmail.com", "sadirijavadi.vahid@gmail.com", msg)
            server.quit()

            
        if not pages_liked_2['data']:
                    
            print('No Page liked')

        else:
                    
            ##### Wrap this block in a while loop so we can keep paginating requests until
            ##### finished.
            while True:

                try:

                    # Perform some action on each post in the collection we receive from
                    # Facebook.
                    [page_liked_query_2(page_liked_2=page_liked) for page_liked in pages_liked_2['data']]
                    # Attempt to make a request to the next page of data, if it exists.
                    pages_liked_2 = requests.get(pages_liked_2['paging']['next']).json()

                except KeyError:
                    print('No More Page!')
                    # When there are no more pages (['paging']['next']), break from the
                    # loop and end the script.
                    break

    #All collected Data for relevant pages as a CSV file 

    #Because of 16 digits problem of Excel:
    relevant_page_ids_modified = []
    all_page_ids_modified = []
    exploring_ids_1_modified = []
    exploring_ids_2_modified = []

    for r_p_i in relevant_page_ids:
        relevant_page_ids_modified.append(str(r_p_i)+'_')

    for a_p_i in all_page_ids:
        all_page_ids_modified.append(str(a_p_i)+'_')

    for e_i_1 in exploring_ids_1:
        exploring_ids_1_modified.append(str(e_i_1)+'_')

    for e_i_2 in exploring_ids_2:
        exploring_ids_2_modified.append(str(e_i_2)+'_')
        
    LIKED_PAGES = Table().with_columns('Relevant Page', relevant_page_ids_modified,
                                       'page_liked_name', page_liked_name,
                                       'page_liked_category', page_liked_category,
                                       'page_liked_about', page_liked_about,
                                       'page_liked_description', page_liked_description,
                                       'page_liked_products', page_liked_products,
                                       'page_liked_link', page_liked_link)

    LIKED_PAGES.to_csv('RelevantPages-'+str(user)+'.csv')    

        
    #All explored IDS as a CSV file
    ALL_LIKED_PAGES = Table().with_column('All_IDs', all_page_ids_modified)
    ALL_LIKED_PAGES.to_csv('AllLikePages-Explored.csv')
    EXPLORING_IDS_1 = Table().with_column('Exploring_IDs_1', exploring_ids_1_modified)
    EXPLORING_IDS_1.to_csv('Exploring_IDs_1.csv')
    EXPLORING_IDS_2 = Table().with_column('Exploring_IDs_2', exploring_ids_2_modified)
    EXPLORING_IDS_2.to_csv('Exploring_IDs_2.csv')

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login("vahid.vsh.68@gmail.com", "vahid9050931")
    msg = "Hi,\n\n Page-liked --->>> Done! For this page: "+str(user)+"\nInformaion of this page is as follows:\n Number of pages for exploring-1: " + str(len(exploring_ids_1_modified))+"\n Number of pages for exploring-2: "+str(len(exploring_ids_2_modified))+"\n Number of relevant pages founded: "+str(len(relevant_page_ids_modified))+"\n Number of all pages explored: "+str(len(all_page_ids_modified))+"\n\n||Vahid S. J.||"
    server.sendmail("vahid.vsh.68@gmail.com", "sadirijavadi.vahid@gmail.com", msg)
    server.quit()

server = smtplib.SMTP('smtp.gmail.com', 587)
server.ehlo()
server.starttls()
server.login("vahid.vsh.68@gmail.com", "vahid9050931")
msg = "Hi,\n\n Page-liked --->>> Done! All relevant pages have been collected!\n\n||Vahid S. J.||"
server.sendmail("vahid.vsh.68@gmail.com", "sadirijavadi.vahid@gmail.com", msg)
server.quit()
