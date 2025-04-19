# import streamlit as st
# from groq import Groq
# import json
# import time
# import os

# # Initialize Groq client with direct API key
# client = Groq(api_key="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")

# def generate_questions(company):
#     """Generate aptitude questions based on the company"""
#     try:
#         completion = client.chat.completions.create(
#             model="llama-3.3-70b-versatile",  # Using the correct model name
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are an expert at creating aptitude test questions. Generate questions that are clear, concise, and have unambiguous answers."
#                 },
#                 {
#                     "role": "user",
#                     "content": f"Generate 10 aptitude questions specific to {company}. Each question should have 4 options. Format the response as a JSON array where each question has 'question', 'options' (array of 4 options), and 'correct_answer' (index of correct option)."
#                 }
#             ],
#             temperature=1,
#             max_tokens=1024,
#             top_p=1,
#             stream=False
#         )
        
#         response_text = completion.choices[0].message.content
#         # Clean the response text to ensure it's valid JSON
#         response_text = response_text.strip()
#         if response_text.startswith('```json'):
#             response_text = response_text[7:]
#         if response_text.endswith('```'):
#             response_text = response_text[:-3]
        
#         questions = json.loads(response_text)
        
#         # Validate the structure
#         if not isinstance(questions, list):
#             st.error("Invalid response format")
#             return None
            
#         for q in questions:
#             if not all(k in q for k in ['question', 'options', 'correct_answer']):
#                 st.error("Invalid question format")
#                 return None
#             if not isinstance(q['options'], list) or len(q['options']) != 4:
#                 st.error("Invalid options format")
#                 return None
                
#         return questions
        
#     except Exception as e:
#         st.error(f"Error generating questions: {str(e)}")
#         return None

# def calculate_score(answers, questions):
#     """Calculate the score based on user answers"""
#     score = 0
#     for i, answer in enumerate(answers):
#         if answer == questions[i]['correct_answer']:
#             score += 10
#     return score

# def main():
#     st.set_page_config(
#         page_title='Aptitude Test',
#         page_icon='📝',
#         layout='wide'
#     )
    
#     st.title('📝 Aptitude Test')
    
#     # Get company from session state
#     if 'company' not in st.session_state:
#         st.error("Please select a company first!")
#         return
    
#     company = st.session_state.company
    
#     # Initialize session state for questions and answers
#     if 'questions' not in st.session_state:
#         st.session_state.questions = None
#     if 'answers' not in st.session_state:
#         st.session_state.answers = []
#     if 'test_completed' not in st.session_state:
#         st.session_state.test_completed = False
    
#     if not st.session_state.questions:
#         with st.spinner('Generating questions...'):
#             st.session_state.questions = generate_questions(company)
#             if not st.session_state.questions:
#                 st.error("Failed to generate questions. Please try again.")
#                 return
    
#     if not st.session_state.test_completed:
#         st.write(f"### Aptitude Test for {company}")
#         st.write("Please answer all questions. Each question carries 10 marks.")
        
#         # Display questions and collect answers
#         for i, q in enumerate(st.session_state.questions):
#             st.write(f"**Question {i+1}:** {q['question']}")
#             answer = st.radio(
#                 "Select your answer:",
#                 q['options'],
#                 key=f"q_{i}",
#                 index=st.session_state.answers[i] if i < len(st.session_state.answers) else 0
#             )
#             if i >= len(st.session_state.answers):
#                 st.session_state.answers.append(q['options'].index(answer))
#             else:
#                 st.session_state.answers[i] = q['options'].index(answer)
        
#         if st.button("Submit Test"):
#             if len(st.session_state.answers) == len(st.session_state.questions):
#                 score = calculate_score(st.session_state.answers, st.session_state.questions)
#                 st.session_state.test_completed = True
#                 st.session_state.score = score
#                 st.success(f"Test completed! Your score: {score}/100")
                
#                 # Store score in session state for main page
#                 st.session_state.aptitude_score = score
                
#                 # Add a delay before redirecting
#                 time.sleep(2)
#                 st.switch_page("App.py")
#             else:
#                 st.error("Please answer all questions before submitting.")
#     else:
#         st.success(f"Test completed! Your score: {st.session_state.score}/100")
#         if st.button("Take Test Again"):
#             st.session_state.questions = None
#             st.session_state.answers = []
#             st.session_state.test_completed = False
#             st.rerun()

# if __name__ == "__main__":
#     main() 

# import streamlit as st
# from groq import Groq
# import json
# import time
# import os
# from dotenv import load_dotenv
# load_dotenv()

# # Initialize Groq client with direct API key
# api_key = os.getenv("GROQ_API_KEY")
# if api_key:
#     print("API Key loaded successfully!")
#     client = Groq(api_key=api_key)
# else:
#     print("API Key NOT loaded.")
#     client = None

# def generate_questions(company):
#     """Generate aptitude questions based on the company"""
#     try:
#         completion = client.chat.completions.create(
#             model="llama-3.3-70b-versatile",  # Using the correct model name
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are an expert at creating aptitude test questions. Generate questions that are clear, concise, and have unambiguous answers."
#                 },
#                 {
#                     "role": "user",
#                     "content": f"Generate 10 aptitude questions related to previous asked questions of {company}. Each question should have 4 options. Format the response as a JSON array where each question has 'question', 'options' (array of 4 options), and 'correct_answer' (index of correct option)."
#                 }
#             ],
#             temperature=1,
#             max_tokens=1024,
#             top_p=1,
#             stream=False
#         )
        
#         response_text = completion.choices[0].message.content
#         # Clean the response text to ensure it's valid JSON
#         response_text = response_text.strip()
#         if response_text.startswith('```json'):
#             response_text = response_text[7:]
#         if response_text.endswith('```'):
#             response_text = response_text[:-3]
        
#         questions = json.loads(response_text)
        
#         # Validate the structure
#         if not isinstance(questions, list):
#             st.error("Invalid response format")
#             return None
            
#         for q in questions:
#             if not all(k in q for k in ['question', 'options', 'correct_answer']):
#                 st.error("Invalid question format")
#                 return None
#             if not isinstance(q['options'], list) or len(q['options']) != 4:
#                 st.error("Invalid options format")
#                 return None
                
#         return questions
        
#     except Exception as e:
#         st.error(f"Error generating questions: {str(e)}")
#         return None

# def calculate_score(answers, questions):
#     """Calculate the score based on user answers"""
#     score = 0
#     for i, answer in enumerate(answers):
#         if answer is not None and answer == questions[i]['correct_answer']:
#             score += 10
#     return score

# def main():
#     st.set_page_config(
#         page_title='Aptitude Test',
#         page_icon='📝',
#         layout='wide'
#     )
    
#     st.title('📝 Aptitude Test')
    
#     # Get company from session state
#     if 'company' not in st.session_state:
#         st.error("Please select a company first!")
#         return
    
#     company = st.session_state.company
    
#     # Initialize session state for questions and answers
#     if 'questions' not in st.session_state:
#         st.session_state.questions = None
#     if 'answers' not in st.session_state:
#         st.session_state.answers = []
#     if 'test_completed' not in st.session_state:
#         st.session_state.test_completed = False
    
#     if not st.session_state.questions:
#         with st.spinner('Generating questions...'):
#             st.session_state.questions = generate_questions(company)
#             if not st.session_state.questions:
#                 st.error("Failed to generate questions. Please try again.")
#                 return
#             # Initialize answers with None values (nothing selected)
#             st.session_state.answers = [None] * len(st.session_state.questions)
    
#     if not st.session_state.test_completed:
#         st.write(f"### Aptitude Test for {company}")
#         st.write("Please answer all questions. Each question carries 10 marks.")
        
#         # Display questions and collect answers
#         for i, q in enumerate(st.session_state.questions):
#             st.write(f"**Question {i+1}:** {q['question']}")
            
#             # Simply use integer index as the selection value, not the text option
#             selected_index = st.radio(
#                 "Select your answer:",
#                 range(len(q['options'])),
#                 key=f"q_{i}",
#                 index=None,  # No default selection
#                 format_func=lambda idx: q['options'][idx]  # Display the option text
#             )
            
#             # Update the answer in session state directly
#             if selected_index is not None:
#                 st.session_state.answers[i] = selected_index
        
#         # Check if all questions are answered
#         all_answered = all(ans is not None for ans in st.session_state.answers)
        
#         if st.button("Submit Test"):
#             if all_answered:
#                 score = calculate_score(st.session_state.answers, st.session_state.questions)
#                 st.session_state.test_completed = True
#                 st.session_state.score = score
#                 st.success(f"Test completed! Your score: {score}/100")
                
#                 # Store score in session state for main page
#                 st.session_state.aptitude_score = score
                
#                 # Add a delay before redirecting
#                 time.sleep(2)
#                 st.switch_page("App.py")
#             else:
#                 st.error("Please answer all questions before submitting.")
#     else:
#         st.success(f"Test completed! Your score: {st.session_state.score}/100")
#         if st.button("Take Test Again"):
#             st.session_state.questions = None
#             st.session_state.answers = []
#             st.session_state.test_completed = False
#             st.rerun()

# if __name__ == "__main__":
#     main()
import streamlit as st
from groq import Groq
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client with API key
api_key = os.getenv("GROQ_API_KEY")
if api_key:
    print("API Key loaded successfully!")
    client = Groq(api_key=api_key)
else:
    print("API Key NOT loaded.")
    client = None

def generate_questions(company):
    """Generate aptitude questions based on the company"""
    try:
        if not client:
            st.error("Groq client is not initialized. Check your API key.")
            return None

        completion = client.chat.completions.create(
            model="llama3-70b-8192",  # Replace with exact model name if needed
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating aptitude test questions. Generate questions that are clear, concise, and have unambiguous answers."
                },
                {
                    "role": "user",
                    "content": f"""
                    Generate 10 aptitude questions based on previously asked questions for {company}.
                    Each question should include:
                    - "question": string
                    - "options": array of 4 strings
                    - "correct_answer": index of correct option (0-based)
                    
                    Return only a valid JSON array. No explanations, markdown, or extra formatting.
                    """
                }
            ],
            temperature=0.7,
            max_tokens=1500,
            top_p=1,
            stream=False
        )

        response_text = completion.choices[0].message.content.strip()

        # Debugging: view the raw response if needed
        print("Raw model response:\n", response_text)

        # Remove markdown-style formatting if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        try:
            questions = json.loads(response_text)
        except json.JSONDecodeError as e:
            st.error(f"JSON decoding error: {e}")
            st.text_area("Raw model response:", value=response_text, height=300)
            return None

        # Validate format
        if not isinstance(questions, list):
            st.error("Invalid response format. Expected a list.")
            return None

        for q in questions:
            if not all(k in q for k in ['question', 'options', 'correct_answer']):
                st.error("One or more questions are missing required fields.")
                return None
            if not isinstance(q['options'], list) or len(q['options']) != 4:
                st.error("Each question must have exactly 4 options.")
                return None
            if not isinstance(q['correct_answer'], int) or not (0 <= q['correct_answer'] < 4):
                st.error("Correct answer index must be between 0 and 3.")
                return None

        return questions

    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return None

def calculate_score(answers, questions):
    """Calculate the score based on user answers"""
    score = 0
    for i, answer in enumerate(answers):
        if answer is not None and answer == questions[i]['correct_answer']:
            score += 10
    return score

def main():
    st.set_page_config(
        page_title='Aptitude Test',
        page_icon='📝',
        layout='wide'
    )

    st.title('📝 Aptitude Test')

    # Ensure company is set
    if 'company' not in st.session_state:
        st.error("Please select a company first!")
        return

    company = st.session_state.company

    # Init session state
    if 'questions' not in st.session_state:
        st.session_state.questions = None
    if 'answers' not in st.session_state:
        st.session_state.answers = []
    if 'test_completed' not in st.session_state:
        st.session_state.test_completed = False

    # Generate questions
    if not st.session_state.questions:
        with st.spinner('Generating questions...'):
            st.session_state.questions = generate_questions(company)
            if not st.session_state.questions:
                st.error("Failed to generate questions. Please try again.")
                return
            st.session_state.answers = [None] * len(st.session_state.questions)

    # Display the test
    if not st.session_state.test_completed:
        st.write(f"### Aptitude Test for {company}")
        st.write("Each question carries 10 marks.")

        for i, q in enumerate(st.session_state.questions):
            st.write(f"**Question {i+1}:** {q['question']}")
            selected_index = st.radio(
                "Select your answer:",
                range(len(q['options'])),
                key=f"q_{i}",
                index=None,
                format_func=lambda idx: q['options'][idx]
            )
            if selected_index is not None:
                st.session_state.answers[i] = selected_index

        all_answered = all(ans is not None for ans in st.session_state.answers)

        if st.button("Submit Test"):
            if all_answered:
                score = calculate_score(st.session_state.answers, st.session_state.questions)
                st.session_state.test_completed = True
                st.session_state.score = score
                st.success(f"Test completed! Your score: {score}/100")
                st.session_state.aptitude_score = score
                time.sleep(2)
                st.switch_page("App.py")  # adjust as needed
            else:
                st.error("Please answer all questions before submitting.")
    else:
        st.success(f"Test completed! Your score: {st.session_state.score}/100")
        if st.button("Take Test Again"):
            st.session_state.questions = None
            st.session_state.answers = []
            st.session_state.test_completed = False
            st.rerun()

if __name__ == "__main__":
    main()
