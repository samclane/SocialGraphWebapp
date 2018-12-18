# DiscordSocialGraph

https://infinite-sands-83078.herokuapp.com/

### Intro

DiscordSocialGraph is my first original Machine Learning project. Originally, I set out to create a Discord bot that would predict the next user to join the server. However, that revealed a more interesting prospect: who is the most popular person on our server? This is an inversion of the original question: which user has the most "draw" to get users to join the server. 

I started by creating an [addon](https://github.com/samclane/Snake-Cogs/blob/master/member_logger/member_logger.py) to the existing Discord Bot framework that I use ([RedBot](https://github.com/Cog-Creators/Red-DiscordBot)). I wanted to collect as little information as possible (to start simple), so all this module does is log 2 things:


1. When a user joins a voice channel, it logs the users in the room before they join

2. When a user mentions another using `@`

Here's what that file looks like:
```
timestamp,member,present
1538687523,194186827534565376,['96323575752962048']
1538687586,196282763555373056,['89841631116681216']
1538698858,89841631116681216,['196282763555373056']
1538703180,96323575752962048,['196282763555373056']
1538703183,194186827534565376,"['196282763555373056', '96323575752962048']"
1538703206,96323575752962048,['202214049117634561']
1538703212,194186827534565376,['202214049117634561']
1538703220,202214049117634561,"['196282763555373056', '96323575752962048', '194186827534565376']"
1538704948,90666704429907968,"['196282763555373056', '96323575752962048', '202214049117634561', '194186827534565376']"
1538704984,90666704429907968,"['196282763555373056', '96323575752962048', '202214049117634561', '194186827534565376']"
1538705291,89841631116681216,"['196282763555373056', '96323575752962048', '202214049117634561', '194186827534565376']"
1538705800,213153711298576394,"['196282763555373056', '96323575752962048', '202214049117634561', '194186827534565376', '89841631116681216']"
1538706849,90666704429907968,"['196282763555373056', '96323575752962048', '194186827534565376', '89841631116681216']"
1538709122,90666704429907968,"['196282763555373056', '96323575752962048', '89841631116681216', '194186827534565376']"
1538712399,202212999509835776,['196282763555373056']
1538759080,202212999509835776,[]
```

The bot collects this information and uploads it to a remote PostgresSQL server. 

The list of all users with interactions on the server is kept as a one-hot vector. The user is treated as the label. The classifier has to use a probabilistic OneVsAll approach, giving a probability distribution over the entire user-base instead of just the top answer. Using this method across the entire userbase generates a distribution of the one-way probability of a user interacting with another user. This will generate a __graph__:

![](https://i.imgur.com/tVC6XqZ.png)

The "popularity" or "draw" of the user is the sum of the weights of all the in-degree weights. Currently, this correlates pretty heavily with the number of instances that user appears in the dataset, but not exactly, meaning that some special relationships are being discovered. For example, it's noticed that `watersnake_test`, my test account, exclusively when I'm already in the server (I only use it when I need to simulate another user besides myself). However, the algorithm can sometimes overfit, drawing strong bonds between my test-account and other accounts I'm actually friends with. It's interesting to try and see the model try and discover who's friends with who. 

Three different models were used in the development of this project:

1. Naive Bayes 
2. Support Vector Machine
3. Multilayer Perceptron

Currently, model #3, the MLP, gives the most accurate results. The accuracy of the model is quantified by the area under the Receiver Operating Characteristic curve. 

![](https://i.imgur.com/eibcGPe.png)

Basically, it describes the correct guesses (True Positive) against the bad guesses (false positive rate) as the threshold for classification is narrowed. 

### The Webapp

Since this application uses information collected from other people, it was suggested to put it online for all to see. The app is hosted on a free Heroku account, with a Hobby Dyno. The heavy ML lifting is done with a RedisQueue background job. The model is retrained from scratch each time it's restarted, as it doesn't take *that* long and gives the most recent, accurate result. 

### Results

From what I've gathered, and from what I can interpret from the metrics returned by the test data set, the classifier is working better than random guessing, by roughly 25%. The way that popularity is calculated could use some work, as it really does heavily on the non-uniform distribution of user participation for the majority of its "Accuracy". Also, some users who have rarely visited the server have really strong bonds with several other users, as a certain user could always be in the server, leading to a 100% bond strength between the outlier and the central user(s). 