import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import spacy
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

final_data = pd.read_csv('final.csv')

nlp = spacy.load('en_core_web_sm')

SIMILARITY_THRESHOLD = 0.01
MIN_QUERY_LENGTH = 1

def preprocess_query(query):
    doc = nlp(query.lower())
    filtered_tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(filtered_tokens)

def get_tfidf_embeddings(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts)


tfidf_vectorizer = TfidfVectorizer()
diagnosis_embeddings = tfidf_vectorizer.fit_transform(final_data['diagnoses'])


def find_best_match(user_input, df):
    user_input = preprocess_query(user_input)
    
    if len(user_input.split()) < MIN_QUERY_LENGTH:
        return None  
    
    user_embedding = tfidf_vectorizer.transform([user_input])
    
    similarities = cosine_similarity(user_embedding, diagnosis_embeddings)
    
    print(f"Similarities: {similarities[0]}")
    
    df['similarity'] = similarities[0]
    
    best_match_index = similarities[0].argmax()
    
    if similarities[0][best_match_index] < SIMILARITY_THRESHOLD:
        return None  
    
    best_match = df.loc[best_match_index]
    return best_match

def process_input():
    user_input = user_entry.get()
    if not user_input:
        messagebox.showerror("Error", "Please enter a symptom or disease.")
        return
    
    result_label.config(text="Processing... Please wait.")
    
    threading.Thread(target=display_result, args=(user_input,)).start()

def display_result(user_input):
    best_match = find_best_match(user_input, final_data)
    
    if best_match is None:
        result_text = "No relevant recommendations found. Please try a different symptom or disease."
    else:
        result_text = f"""
        Drug: {best_match['drug']}
        Formulary Drug Code: {best_match['formulary_drug_cd']}
        Product Strength: {best_match['prod_strength']}
        Form RX: {best_match['form_rx']}
        Dose Value RX: {best_match['dose_val_rx']}
        Dose Unit RX: {best_match['dose_unit_rx']}
        Doses per 24 hrs: {best_match['doses_per_24_hrs']}
        Route: {best_match['route']}
        Procedures: {best_match['procedures']}
        Description: {best_match['description']}
        """
    
    result_label.after(0, result_label.config, {'text': result_text})

root = tk.Tk()
root.title("Drug Recommendations")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

ttk.Label(frame, text="Enter symptom or disease:").grid(row=0, column=0, sticky=tk.W)
user_entry = ttk.Entry(frame, width=50)
user_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))

process_button = ttk.Button(frame, text="Get Recommendations", command=process_input)
process_button.grid(row=1, column=0, columnspan=2)

result_label = ttk.Label(frame, text="", wraplength=400)
result_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

root.mainloop()
