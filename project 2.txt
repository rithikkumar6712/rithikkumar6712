import spacy
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample documents for plagiarism checking
original_text = "The quick brown fox jumps over the lazy dog."
suspicious_text = "A fast brown fox jumps over a lazy dog."

# Function to preprocess the text and extract word embeddings
def preprocess_and_embed(text):
    tokens = nlp(text.lower())
    return [token.vector for token in tokens if not token.is_stop and token.is_alpha]

# Preprocess and embed the texts
original_embedded = preprocess_and_embed(original_text)
suspicious_embedded = preprocess_and_embed(suspicious_text)

# Convert lists of embeddings to numpy arrays
original_embedded = np.array(original_embedded)
suspicious_embedded = np.array(suspicious_embedded)

# Calculate cosine similarity between the embeddings
similarity_score = cosine_similarity([original_embedded.mean(axis=0)], [suspicious_embedded.mean(axis=0)])[0][0]

# Set a similarity threshold (you can adjust this based on your needs)
similarity_threshold = 0.9

# Check for plagiarism
if similarity_score >= similarity_threshold:
    print("Plagiarism detected!")
    print(f"Similarity Score: {similarity_score:.2f}")
else:
    print("No plagiarism detected.")
    print(f"Similarity Score: {similarity_score:.2f}")


output :

AI Plagiarism Checker - Report

Document Checked: [Document Title]

Date: [Date of Checking]

Summary:
The AI plagiarism checker has analyzed the provided document for potential plagiarism and has generated a detailed report highlighting the detected similarities with external sources. The analysis indicates the presence of both direct matches and paraphrased content. Below is a breakdown of the results:

Overall Similarity Score: [X%] (Percentage of text similarity detected)

Direct Matches:

Source: [Source Title or URL]