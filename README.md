# What drives changes in attitudes towards refugees? <br /> Evidence from Social Media #
During the “refugee crisis”, Facebook was blamed for providing a platform for hate speech, intoxicating the public debate on migrants in general and in particular refugees. Politicians pointed at its algorithms for creating echo chambers that aggravated political polarization and marginalized moderate views. While the call for changes to the management of social media communication is widespread, there is little systematic evidence about the extent and the way social-media-users voice negative or positive sentiments towards refugees and migrants. What topics do they talk about when expressing such strong attitudes? And how do they respond to drastic events, such as the ‘Kölner Silvesternacht’, the New Year’s Eve Events in Cologne, Germany?

Drawing on a novel dataset based on Facebook comments to newspaper articles on refugees and migrants, we study the impact of such drastic events on social media users’ pro- or anti-refugee sentiments. Moreover, we identify the topics discussed in the context of these sentiments, e.g. economic issues, legal issues, or fear and anger. Our dataset has been compiled using a supervised machine-learning algorithm trained on several thousand manually annotated Facebook comments. It provides unique insights into public debates held on social media platforms. With this extraordinarily rich dataset, we assess how social media users’ sentiments towards refugees and migrants have developed over time. Articles’ and comments’ time stamps also allow us to investigate whether sentiments are driven by specific events and if so, how this impact varies across newspaper outlets and across regions in Germany. The paper thereby contributes to our understanding of how social media users communicate about politically relevant themes in times of crisis, and informs the debate on how sentiments towards minority groups evolve.

This Master Thesis was designed and developed as a project to:

**1. Store**<br />
**2. Retrieve**<br />
**3. Visualize**<br />
**4. Communicate Data**

Main Targets of the project were:

**•	Data Collection and Storage:** Collecting relevant Facebook Comments, Post Descriptions, Attachments, Likes, etc. using Python<br />
**• Recognition of relevant Facebook Pages:** Use of Network Nodes (Liked Pages by Pages) to recognize relevant Facebook Pages<br />
**• Data Pre-Processing and Cleansing:** Design of Spelling Corrector using Python<br />
**• Data Annotation:** Group Member of Instructors for Student Assistants to annotate Comments and Post Descriptions<br />
**• Sentiment Analysis:** Classification using Doc2vec and Supervised Machine Learning Algorithms in Python<br />
**• Topic Extraction:** Topic Prediction for Post and Attachment Description using LDA in PythonQuantifying the Qualitative Data<br />
**•**	Comparing the sentiments per state, city and gender of users as well as time of the comments<br />
**•**	Visualizing and Communicating the Results

### Gathering Data ###

We decided to look at the 85 Facebook pages of newspapers. For each Facebook page, we recorded the following information:

**1.	Facebook Post – ID<br />
2.	Facebook Post – Type (Link, Photo, Video, Status)<br />
3.	Facebook Post – Created Time<br />
4.	Facebook Post – Description<br />
5.	Post Reactions – Like-/ Angry-/ Love-/ Wow-/ Sad-/ Haha-Count<br />
6.	Post Attachment – Title<br />
7.	Post Attachment – Description<br />
8.	Post Attachment – Link<br />
9.	Post Comment – ID<br />
10.	Post Comment – Created Time<br />
11.	Post Comment – User ID<br />
12.	Post Comment – User Name<br />
13.	Post Comment – Message<br />
14.	Post Comment – Like<br />
15.	Post Comment – Comment<br />
16.	Post Comment Comment – Like<br />
17.	Facebook User – Profile Picture**

In the following figure, the process of data collection has been showed:

![Gathering Data](https://github.com/Vahidsj/ProjectWork-IfW/blob/master/Image/Gathering%20Data.png)

### Data Pipeline: ###

The visual below is an overview of our data pipeline for this project:

![Data Pipeline](https://github.com/Vahidsj/ProjectWork-IfW/blob/master/Image/Data%20Pipeline.png)
