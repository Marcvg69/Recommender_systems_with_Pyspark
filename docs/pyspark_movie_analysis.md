# Movie Recommendation System

- Repository: `Recommender systems with Pyspark`
- Type of Challenge: `Learning`
- Duration: `5 days`
- Deadline: `19/09/2025 16:30 PM`
- Challenge: Individual 

## The mission

You are part of the data team for the Internet Movie Database (IMDB). Your task is to create a recommendation tool for movies and TV shows that an user can interact with. 

The tool will take as input the user's favorite movies and shows and recommend new ones in a user friendly manner. You are allowed to use additional resources and incorporate other features that are interesting for users.

The goal is to have a working minimum viable product (MVP) by the end of the project. Each member (or team) in the group should submit a project outline of their MVP on by end of Day 1 to the coach for review and feedback. 

### The data

* [MovieLens](https://grouplens.org/datasets/movielens/). A subset of this data can be found in [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).
* IMDB provides a public dataset that can be used for this purpose: [IMDB dataset](https://www.imdb.com/interfaces/). You can also decide to scrape your own data or supplement it with other sources. 

### The model 

There are three especially common methods to design a recommender system: Collaborative Filtering, Content-based Filtering and hybrid techniques. Content-based systems aim to make recommendations based on some previous information about the customer and the products. For instance, if Netflix knows you like drama movies, it might recommend you movies of this type. 

However, in a collaborative filtering approach, it might simply ignore the type of the film. The features used in this case are exclusively users rating patterns. For instance, if you watched five different series on Netflix and have rated five each of them, just like some other random user, then you might be interested to know what else he has rated as five stars. Hybrid systems make use of both techniques.

In this project, we will use of Alternating Least Square (ALS) Matrix Factorization, a Collaborative Filtering algorithm that is already implemented in the ML Pyspark library. 

### Goal
As data engineers, you will be tasked to create a end-to-end product for users. Your focus is on the data pipeline and piecing together some of the tools you've learned so far to consolidate your knowledge.  

### Mission objective:

* Create a simple movie recommendation app using Pyspark in the backend. 
* Containerize the scripts with Docker and/or Kubernetes (optional).
* Deploy app locally or on Render (streamlit recommended!). 
* (Optional) Orchestrate updates to pipeline using Airflow. 
* (Optional) Create a database to store and update data recommendations for users.

### Resources on PySpark:

* [Movie Recommendation System Using Spark MLlib](https://medium.com/edureka/spark-mllib-e87546ac268) (See this article for a sample pipeline)
* [Pyspark Tutorial: Getting Started with Pyspark](https://www.datacamp.com/tutorial/pyspark-tutorial-getting-started-with-pyspark)
* [Building Recommendation Engines with PySpark (Chapter 3)](https://www.datacamp.com/courses/recommendation-engines-in-pyspark)

### Tutorials about recommendation systems (General)

* [Recommender systems in Python ](https://www.datacamp.com/tutorial/recommender-systems-python)
* [Recommender Systems Python-Methods and Algorithms](https://www.projectpro.io/article/recommender-systems-python-methods-and-algorithms/413)
* [A Complete Guide To Recommender Systems — Tutorial with Sklearn, Surprise, Keras, Recommenders](https://towardsdatascience.com/a-complete-guide-to-recommender-system-tutorial-with-sklearn-surprise-keras-recommender-5e52e8ceace1)
* [Top 3 Python Packages to Learn the Recommendation System](https://towardsdatascience.com/top-3-python-package-to-learn-the-recommendation-system-bb11a916b8ff)

## Deliverables

* Publish your source code on the GitHub repository.
* README and repository is complete and well documented
* Small presetation of results (10 minutes + 5 Q&A)
    - Pimp up the README file:
    - Description
    - Installation
    - Usage
    -⚠️ DATA SOURCES
    - (Visuals)
    - (Contributors)
    - (Timeline)
    (Personal situation)
    
## Evaluation criteria

### Technical
- Publish clean and readable code on GitHub.
- README has the format specified in the #Deliverables section?
- Use of functions and good coding practices
- You have a well-define pipeline that runs without errors.

## You've got this!

![You've got this!](https://media.giphy.com/media/KZe02gpoAj4yVjxKQt/giphy.gif)
