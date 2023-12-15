import nltk
import spacy
import simcse
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def tf_idf_filter(t_or_w, tutorial, threshold):
  tutorial_vect = TfidfVectorizer()
  tutorial_tfidf_matrix = tutorial_vect.fit_transform([tutorial])

  t_or_w_vect = TfidfVectorizer()
  t_or_w_tfidf_matrix = t_or_w_vect.fit_transform([t_or_w])
  df = pd.DataFrame(t_or_w_tfidf_matrix.toarray(), columns = t_or_w_vect.get_feature_names())

  try:
    tokens = [token for token in t_or_w_vect.get_feature_names() if (token in tutorial_vect.get_feature_names()) and (token not in stop_words)]
    tfidf_scores = [df[token][0] for token in tokens]
    if max(tfidf_scores) >= threshold:
      return True
  except:
    return False
  
  return False

def length_filter(text, tokenizer, max_threshold, min_threshold):
  return (len(tokenizer.tokenize(text)) > min_threshold) and (len(tokenizer.tokenize(text)) < max_threshold)

def lexical_overlap_filter(positive_candidate, negative_candidates, threshold):
  positive_candidate = set([token.lemma_ for token in lemmatizer(positive_candidate) if token.lemma_ not in stop_words])

  for negative_candidate in negative_candidates:
    negative_candidate = set([token.lemma_ for token in lemmatizer(negative_candidate) if token.lemma_ not in stop_words])
    if len(negative_candidate.intersection(positive_candidate)) / len(negative_candidate.union(positive_candidate)) >= threshold:
      return False

  return True

def similarity_filter(negative_candidates, gold_t_or_ws, simcse, threshold):
  for negative_candidate in negative_candidates:
    similarities = simcse.similarity([negative_candidate], gold_t_or_ws).tolist()[0]
    if (max(similarities) >= threshold):
      return False
          
  return True

def similarity_filter_for_pos_neg_candidates(negative_candidates, positive_candidate, simcse, threshold):
  negative_candidates = [candidate.replace("How to", "") for candidate in negative_candidates]
  positive_candidate = positive_candidate.replace("How to", "")

  similarities = simcse.similarity([positive_candidate], negative_candidates).tolist()[0]
  if (max(similarities) >= threshold):
    return False
  
  return True

def similarity_filter_reverse(negative_candidates, t_or_ws, t_or_w_inference, simcse, threshold):
  for negative_candidate in negative_candidates:
    t_or_ws_of_negative_candidate = t_or_w_inference[negative_candidate]
    for t_or_w in t_or_ws:
      similarities = simcse.similarity([t_or_w], t_or_ws_of_negative_candidate).tolist()[0]
      if (max(similarities) >= threshold):
        return False
    
  return True

def category_filter(category_list):
  unwanted_main_categories = ["Philosophy and Religion", "Holidays and Traditions"]
  unwanted_sub_categories = ["Fandom", "Celebrities", "Calculators", "Pokemon Video Games", "Minecraft", "Trading Card Games", "Alternative Health", "Games", "Rocks and Minerals"] 
  unwanted_categories = unwanted_main_categories + unwanted_sub_categories
  for category in category_list:
    if category in unwanted_categories:
      return False
    
  return True

def source_tutorial_filter(positive_candidate, negative_candidates, source_file):
  positive_candidate_tutorial = [k for k, v in source_file.items() if positive_candidate in v][0]
  negative_candidate_tutorials = []

  for negative_candidate in negative_candidates:
    negative_candidate_tutorial = [k for k, v in source_file.items() if negative_candidate in v][0]
    negative_candidate_tutorials.append(negative_candidate_tutorial)
  
  source_tutorials = [positive_candidate_tutorial] + negative_candidate_tutorials
  
  if len(set(source_tutorials)) == 4:
    return True
  else:
    return False