
import streamlit as st
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os
import json
import requests
import pandas as pd
from ast import keyword
import re
# Get API key from user data
google_api_key = "AIzaSyD0J1qbVaIEI0Lfah8knilG9aKHvxJ8rzo"
os.environ["GOOGLE_API_KEY"] = google_api_key
genai.configure(api_key=google_api_key)

# --- Functions ---
def creat_seo_prompt(tweet):
    template = """ Act as an SEO expert. I will give you a tweet. Extract the important keywords from the tweet for SEO purposes and output them in JSON format with keywords as the key and relevance score as the value. \
    Do not include any headers or footers in the output.
    Tweet: {tweet} """

    prompt = PromptTemplate(input_variables=["tweet"], template=template)
    seo_prompt = prompt.format(tweet=tweet)
    return seo_prompt

def find_seo_keywords(tweet):
    seo_prompt = creat_seo_prompt(tweet)
    response = genai.generate_text(
        model="models/text-bison-001",
        prompt=seo_prompt,
        temperature=0.2,
        max_output_tokens=256
    )
    return response.result

def generate_query(keyword):
    keyword = json.loads(keyword)
    query = list(keyword.keys())
    query = ' '.join(query) + ' -filter:retweets'
    return query

def get_similar_tweets(query):
    url = "https://twitter154.p.rapidapi.com/search/search/continuation"

    querystring = {
        "query": query,
        "section": "top",
        "min_retweets": "20",
        "limit": "5",
        "continuation_token": "DAACCgACF_Sz76EAJxAKAAMX9LPvoP_Y8AgABAAAAAILAAUAAABQRW1QQzZ3QUFBZlEvZ0dKTjB2R3AvQUFBQUFVWDlJWmx4cHZBZkJmMG5RNUxHdUVQRi9TdTZPSGJzQ0VYOUp6Y3psdUJ3UmYwbFE3Q1dxQWsIAAYAAAAACAAHAAAAAAwACAoAARf0hmXGm8B8AAAA",
        "min_likes": "20",
        "start_date": "2022-01-01",
        "language": "en"
    }

    headers = {
        "x-rapidapi-key": "db28b527abmshb30d96d7e746dc0p111f55jsnc84d27ff4891",
        "x-rapidapi-host": "twitter154.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    similar_tweets = pd.DataFrame(response.json()["results"])
    return similar_tweets

def generate_blog_from_tweets(tweets):
    tweet_content = "\n".join(f"- {tweet}" for tweet in tweets)
    prompt = (
        "Given the following tweets, create a well-structured and coherent blog post:\n\n"
        f"{tweet_content}\n\n"
        "The blog post should be engaging, detailed, and suitable for publication on a website."
    )
    response = genai.generate_text(
        model="models/text-bison-001",
        prompt=prompt,
        temperature=0.2,
        max_output_tokens=256
    )
    return response

def get_image_url(data):
    count = 0
    urls = []

    for i in data:
        if count>2:
            break
        else:
            if len(i)!=0:
                for j in i:
                    if re.search(".jpg",j):
                        urls.append(j)
                        count += 1
    return urls

# --- Streamlit App ---
st.title("Tweet to Blog Post Generator")

tweet = st.text_area("Enter your tweet:")

if st.button("Find SEO Keywords"):
    if tweet:
        keywords_json = find_seo_keywords(tweet)
        keywords_dict = json.loads(keywords_json)
        keywords_df = pd.DataFrame.from_dict(keywords_dict, orient='index', columns=['Relevance Score'])
        st.write("## SEO Keywords")
        st.table(keywords_df)
    else:
        st.warning("Please enter a tweet first.")

if st.button("Find Similar Tweets"):
    if tweet:
        keywords_json = find_seo_keywords(tweet)
        query = generate_query(keywords_json)
        similar_tweets = get_similar_tweets(query)
        st.write("## Similar Tweets")
        st.table(similar_tweets)
    else:
        st.warning("Please find SEO keywords first.")

if st.button("Generate Blog Post"):
    if tweet:
        keywords_json = find_seo_keywords(tweet)
        query = generate_query(keywords_json)
        similar_tweets = get_similar_tweets(query)
        blog_post = generate_blog_from_tweets(similar_tweets["text"]).result
        images = get_image_url(similar_tweets["media_url"])
        st.write("## Blog Post")
        st.write(blog_post)
        col1, col2 = st.columns(2)
        with col1:
            if images[0]:
                st.image(images[0])
        with col2:
            if images[1]:
                st.image(images[1])
    else:
        st.warning("Please find similar tweets first.")
