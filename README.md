# Reinforcement-Learning on Retail

### * RL in Recommendation System

#### PROBLEM STATEMENT :
  Traditional recommendation task can be treated as sequential decision making problem. Recommender interacts with users to sequentially suggest set of items. The goal here is to maximize clients' satisfaction (i.e. reward) by using Reinforcement Learning to suggest more personalized and new contents to the user to develop interest on other products and thus increase the sales of the products.
  
##### Reinforcement Learning can help recommendation at least in 2 ways.
  
  1. User’s preference on previous items will affect his choice on the next items. User tends to give a higher rating if he has consecutively received more satisfied items (and vice versa). So, it would be more reasonable to model the recommendation as a sequential decision making process.

  2. It is important to use long-term planning in recommendations. For example, after reading the weather forecast, the user is not willing to read similar news. On the other hand, after watching funny videos or reading memes the user can constantly do the same.
  
  
  #### Movielens (1M) dataset results

| Model                          | nDCG@10 | hit_rate@10 |
| ------------------------------ | :-----: | ----------- |
| **DDPG with OU noise**         |  0.280  | 0.502       |
| DDPG                           |  0.254  | 0.454       |
| Neural Collaborative Filtering |  0.238  | 0.430       |
| Random (for comparison)        |  ~0.05  | ~0.1        |
  
- RL in NLP
- RL in Supply Chain
- RL in Retail
