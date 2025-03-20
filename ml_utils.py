import logging
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

def extract_skills(text, skill_classifier=None):
    """Extract skills from resume text"""
    if not text:
        return []
        
    # Define a comprehensive list of technical skills
    technical_skills = [
        "python", "java", "javascript", "html", "css", "react", "angular",
        "node.js", "express", "django", "flask", "sql", "mysql", "mongodb",
        "postgresql", "aws", "azure", "docker", "kubernetes", "git",
        "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn",
        "pandas", "numpy", "data analysis", "power bi", "tableau",
        "c++", "c#", "php", "ruby", "swift", "kotlin", "flutter",
        "android", "ios", "react native", "vue.js", "typescript",
        "jenkins", "ci/cd", "agile", "scrum", "jira"
    ]
    
    found_skills = []
    text = text.lower()
    
    # Look for each skill in the text
    for skill in technical_skills:
        if skill in text:
            found_skills.append(skill)
    
    # Remove duplicates and sort
    found_skills = list(set(found_skills))
    found_skills.sort()
    
    return found_skills

def predict_domain(text, domain_predictor=None):
    """Predict domain from resume text"""
    if not text:
        return "Unknown"
    
    text = text.lower()
    domains = {
        'Data Science': ['python', 'machine learning', 'data analysis', 'statistics', 
                        'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy'],
        'Web Development': ['html', 'css', 'javascript', 'react', 'node', 'django', 
                          'flask', 'php', 'angular'],
        'Android Development': ['android', 'kotlin', 'java', 'mobile development', 
                              'flutter', 'react native'],
        'IOS Development': ['ios', 'swift', 'objective-c', 'xcode'],
        'UI-UX Development': ['ui', 'ux', 'figma', 'adobe xd', 'sketch', 'design']
    }
    
    # Count matches for each domain
    domain_scores = {}
    for domain, keywords in domains.items():
        score = sum(1 for keyword in keywords if keyword in text)
        domain_scores[domain] = score
    
    # Return domain with highest score, or "General" if no matches
    if max(domain_scores.values()) > 0:
        return max(domain_scores.items(), key=lambda x: x[1])[0]
    return "General"

def train_models_if_needed():
    """Placeholder for model training"""
    # This is a simplified version that returns None for both models
    # The actual implementation would load or train ML models
    return None, None

# Initialize NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.error(f"Failed to download NLTK resources: {str(e)}")

#     import logging
# import nltk
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import Pipeline
# import joblib
# import os

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# # Constants
# MODEL_DIR = "models"
# SKILL_MODEL_PATH = os.path.join(MODEL_DIR, "skill_classifier.pkl")
# DOMAIN_MODEL_PATH = os.path.join(MODEL_DIR, "domain_predictor.pkl")

# # Ensure model directory exists
# os.makedirs(MODEL_DIR, exist_ok=True)

# def extract_skills(text, skill_classifier=None):
#     """Extract skills from resume text using a trained ML model."""
#     if not text:
#         return []
    
#     if skill_classifier is None:
#         # Fallback to rule-based approach if no model is provided
#         return extract_skills_rule_based(text)
    
#     # Use the ML model to predict skills
#     skills = skill_classifier.predict([text])
#     return list(set(skills))

# def extract_skills_rule_based(text):
#     """Fallback rule-based skill extraction."""
#     technical_skills = [
#         "python", "java", "javascript", "html", "css", "react", "angular",
#         "node.js", "express", "django", "flask", "sql", "mysql", "mongodb",
#         "postgresql", "aws", "azure", "docker", "kubernetes", "git",
#         "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn",
#         "pandas", "numpy", "data analysis", "power bi", "tableau",
#         "c++", "c#", "php", "ruby", "swift", "kotlin", "flutter",
#         "android", "ios", "react native", "vue.js", "typescript",
#         "jenkins", "ci/cd", "agile", "scrum", "jira"
#     ]
    
#     found_skills = []
#     text = text.lower()
#     for skill in technical_skills:
#         if skill in text:
#             found_skills.append(skill)
#     return list(set(found_skills))

# def predict_domain(text, domain_predictor=None):
#     """Predict domain from resume text using a trained ML model."""
#     if not text:
#         return "Unknown"
    
#     if domain_predictor is None:
#         # Fallback to rule-based approach if no model is provided
#         return predict_domain_rule_based(text)
    
#     # Use the ML model to predict the domain
#     domain = domain_predictor.predict([text])[0]
#     return domain

# def predict_domain_rule_based(text):
#     """Fallback rule-based domain prediction."""
#     domains = {
#         'Data Science': ['python', 'machine learning', 'data analysis', 'statistics', 
#                         'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy'],
#         'Web Development': ['html', 'css', 'javascript', 'react', 'node', 'django', 
#                           'flask', 'php', 'angular'],
#         'Android Development': ['android', 'kotlin', 'java', 'mobile development', 
#                               'flutter', 'react native'],
#         'IOS Development': ['ios', 'swift', 'objective-c', 'xcode'],
#         'UI-UX Development': ['ui', 'ux', 'figma', 'adobe xd', 'sketch', 'design']
#     }
    
#     domain_scores = {}
#     for domain, keywords in domains.items():
#         score = sum(1 for keyword in keywords if keyword in text.lower())
#         domain_scores[domain] = score
    
#     if max(domain_scores.values()) > 0:
#         return max(domain_scores.items(), key=lambda x: x[1])[0]
#     return "General"

# def train_models_if_needed():
#     """Load or train ML models for skill extraction and domain prediction."""
#     # Load models if they exist
#     if os.path.exists(SKILL_MODEL_PATH) and os.path.exists(DOMAIN_MODEL_PATH):
#         skill_classifier = joblib.load(SKILL_MODEL_PATH)
#         domain_predictor = joblib.load(DOMAIN_MODEL_PATH)
#         logging.info("Loaded pre-trained models.")
#     else:
#         # Train new models (placeholder for actual training logic)
#         logging.info("Training new models...")
#         skill_classifier, domain_predictor = train_models()
#         joblib.dump(skill_classifier, SKILL_MODEL_PATH)
#         joblib.dump(domain_predictor, DOMAIN_MODEL_PATH)
#         logging.info("Models trained and saved.")
    
#     return skill_classifier, domain_predictor

# def train_models():
#     """Placeholder for actual model training logic."""
#     # Example: Train a RandomForestClassifier for domain prediction
#     # You would replace this with actual training code using a labeled dataset
#     vectorizer = TfidfVectorizer()
#     classifier = RandomForestClassifier()
#     domain_predictor = Pipeline([("tfidf", vectorizer), ("clf", classifier)])
    
#     # Dummy training data (replace with real data)
#     X_train = ["python machine learning", "html css javascript", "android kotlin"]
#     y_train = ["Data Science", "Web Development", "Android Development"]
    
#     domain_predictor.fit(X_train, y_train)
    
#     # For skill extraction, you would train a separate model or use a pre-trained NER model
#     skill_classifier = None  # Placeholder
    
#     return skill_classifier, domain_predictor

# # Initialize NLTK
# try:
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)
# except Exception as e:
#     logging.error(f"Failed to download NLTK resources: {str(e)}")