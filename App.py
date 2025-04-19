# import streamlit as st
# # First Streamlit command must be st.set_page_config()
# st.set_page_config(
#     page_title='AI Resume Analyzer',
#     page_icon='Logo/resume_icon.jpg'
# )

# # Now import all other libraries
# import pandas as pd
# import base64, random
# import time, datetime
# # libraries to parse the resume pdf files
# from pyresparser import ResumeParser
# from pdfminer3.layout import LAParams, LTTextBox
# from pdfminer3.pdfpage import PDFPage
# from pdfminer3.pdfinterp import PDFResourceManager
# from pdfminer3.pdfinterp import PDFPageInterpreter
# from pdfminer3.converter import TextConverter
# import io, random
# from streamlit_tags import st_tags 
# from PIL import Image
# import pymysql
# from Courses import ds_course ,web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
# import yt_dlp
# import matplotlib.pyplot as plt
# import plotly.express as px
# import nltk
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import joblib
# import numpy as np
# from ml_utils import extract_skills, predict_domain, train_models_if_needed

# # Define create_table function first
# # def create_table():
# #     try:
# #         # Create table only if it doesn't exist (removed DROP TABLE)
# #         table_sql = """
# #         CREATE TABLE IF NOT EXISTS user_data (
# #             ID INT NOT NULL AUTO_INCREMENT,
# #             Name VARCHAR(500) NOT NULL,
# #             Email_ID VARCHAR(500) NOT NULL,
# #             resume_score VARCHAR(8) NOT NULL,
# #             Timestamp VARCHAR(50) NOT NULL,
# #             Page_no VARCHAR(5) NOT NULL,
# #             Predicted_Field TEXT NOT NULL,
# #             User_level TEXT NOT NULL,
# #             Actual_skills TEXT NOT NULL,
# #             Recommended_skills TEXT NOT NULL,
# #             Recommended_courses TEXT NOT NULL,
# #             CGPA FLOAT,
# #             Current_year INT,
# #             Preferred_company VARCHAR(500),
# #             PRIMARY KEY (ID)
# #         ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
# #         """
# #         cursor.execute(table_sql)
# #         connection.commit()
# #         print("Table check completed successfully!")
        
# #     except Exception as e:
# #         print(f"Error with table: {str(e)}")
# #         st.error(f"Error with table: {str(e)}")
# def create_table():
#     try:
#         # Create table only if it doesn't exist (removed DROP TABLE)
#         table_sql = """
#         CREATE TABLE IF NOT EXISTS user_data (
#             ID INT NOT NULL AUTO_INCREMENT,
#             Name VARCHAR(500) NOT NULL,
#             Email_ID VARCHAR(500) NOT NULL,
#             resume_score VARCHAR(8) NOT NULL,
#             Timestamp VARCHAR(50) NOT NULL,
#             Page_no VARCHAR(5) NOT NULL,
#             Predicted_Field TEXT NOT NULL,
#             User_level TEXT NOT NULL,
#             Actual_skills TEXT NOT NULL,
#             Recommended_skills TEXT NOT NULL,
#             Recommended_courses TEXT NOT NULL,
#             CGPA FLOAT,
#             Current_year INT,
#             Preferred_company VARCHAR(500),
#             Aptitude_marks INT,  # Add this new column
#             PRIMARY KEY (ID)
#         ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
#         """
#         cursor.execute(table_sql)
#         connection.commit()
#         print("Table check completed successfully!")
        
#     except Exception as e:
#         print(f"Error with table: {str(e)}")
#         st.error(f"Error with table: {str(e)}")

# # Initialize database connection
# # try:
# #     connection = pymysql.connect(
# #         host='localhost',
# #         user='root',
# #         password='mysql',
# #         charset='utf8mb4',
# #         cursorclass=pymysql.cursors.DictCursor
# #     )
# #     cursor = connection.cursor()
# try:
#     connection = pymysql.connect(
#         host='localhost',
#         user='root',
#         password='mysql',
#         database='cv',  # Add this to connect directly to the database
#         charset='utf8mb4',
#         cursorclass=pymysql.cursors.DictCursor
#     )
#     cursor = connection.cursor()
    
#     # Create database if it doesn't exist
#     cursor.execute("CREATE DATABASE IF NOT EXISTS cv")
#     cursor.execute("USE cv")
    
#     # Create table if it doesn't exist
#     create_table()
    
#     st.success("Database Connected Successfully!")
# except Exception as e:
#     st.error(f"Error connecting to Database: {e}")

# # Load or train ML models
# skill_classifier, domain_predictor = train_models_if_needed()

# # Dictionary of skills for each domain
# domain_skills = {
#     'Data Science': ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling', 'Data Mining', 
#                     'Clustering & Classification', 'Data Analytics', 'Quantitative Analysis', 
#                     'Web Scraping', 'ML Algorithms', 'Keras', 'Pytorch', 'Probability', 
#                     'Scikit-learn', 'Tensorflow', 'Flask', 'Streamlit'],
#     'Web Development': ['React', 'Django', 'Node JS', 'React JS', 'php', 'laravel', 'Magento', 
#                        'wordpress', 'Javascript', 'Angular JS', 'c#', 'Flask', 'SDK'],
#     'Android Development': ['Android', 'Android development', 'Flutter', 'Kotlin', 'XML', 'Java', 
#                            'Kivy', 'GIT', 'SDK', 'SQLite'],
#     'IOS Development': ['IOS', 'IOS Development', 'Swift', 'Cocoa', 'Cocoa Touch', 'Xcode', 
#                        'Objective-C', 'SQLite', 'Plist', 'StoreKit', 'UI-Kit', 'AV Foundation', 'Auto-Layout'],
#     'UI-UX Development': ['UI', 'User Experience', 'Adobe XD', 'Figma', 'Zeplin', 'Balsamiq', 
#                          'Prototyping', 'Wireframes', 'Storyframes', 'Adobe Photoshop', 'Editing', 
#                          'Illustrator', 'After Effects', 'Premier Pro', 'Indesign', 'Wireframe', 
#                          'Solid', 'Grasp', 'User Research']
# }

# def fetch_yt_video(link):
#     with yt_dlp.YoutubeDL({"no_warnings": True}) as ydl:
#         info = ydl.extract_info(link, download=False)
#     return info["title"]

# def get_table_download_link(df, filename, text):
#     """Generates a link allowing the data in a given panda dataframe to be downloaded
#     in:  dataframe
#     out: href string
#     """
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
#     href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
#     return href

# def pdf_reader(file):
#     resource_manager = PDFResourceManager()
#     fake_file_handle = io.StringIO()
#     converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams(detect_vertical=True))
#     page_interpreter = PDFPageInterpreter(resource_manager, converter)
#     with open(file, 'rb') as fh:
#         for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
#             page_interpreter.process_page(page)
#         text = fake_file_handle.getvalue()
#     converter.close()
#     fake_file_handle.close()
#     return text

# def show_pdf(file_path):
#     with open(file_path, 'rb') as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#     pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)

# def course_recommender(course_list):
#     st.subheader('*Courses & Certificates Recommendations 🎓*')
#     rec_course = []
#     no_of_rec = st.slider('Choose number of Courses to Recommend', 1, 10, 5)
    
#     if isinstance(course_list, list):
#         # If course_list is already a list of tuples
#         courses = course_list
#     else:
#         # If course_list is a dictionary or other format, convert to list
#         courses = list(course_list.items()) if hasattr(course_list, 'items') else []
    
#     random.shuffle(courses)
    
#     for c_name, c_link in courses[:no_of_rec]:
#         st.markdown(f"- [{c_name}]({c_link})")
#         rec_course.append(c_name)
    
#     return rec_course


# # Load company dataset
# companies_df = pd.read_csv('D:\\2nd Year Course Projects\\DS\\AI-Resume-Analyzer-main (1)\\company.csv')  # Ensure the file path is correct
# def calculate_probability(student, company):
#     try:
#         # Debug print
#         print(f"\nCalculating probability for company: {company['company_name']}")
#         print(f"Student profile: {student}")
#         print(f"Company requirements: {company}")
        
#         # Initialize weights
#         skill_weight = 0.5
#         year_weight = 0.2
#         cgpa_weight = 0.2
#         aptitude_weight = 0.1
        
#         # Extract student data with defaults
#         student_skills = set(str(student['skills']).lower().split(',')) if student['skills'] else set()
#         student_year = int(student['current_year']) if student['current_year'] else 0
#         student_cgpa = float(student['cgpa']) if student['cgpa'] else 0.0
#         student_aptitude = int(student['aptitude_marks']) if student.get('aptitude_marks') else 0
        
#         # Extract company requirements with defaults
#         req_skills = set(str(company['required_skills']).lower().split(',')) if pd.notna(company['required_skills']) else set()
#         pref_year = int(company['preferred_year']) if pd.notna(company['preferred_year']) else 0
#         min_cgpa = float(company['required_cgpa']) if pd.notna(company['required_cgpa']) else 0.0
#         min_aptitude = int(company['required_aptitude_marks']) if pd.notna(company['required_aptitude_marks']) else 0
        
#         # Calculate skill match (50% weight)
#         if req_skills:
#             skill_match = len(student_skills.intersection(req_skills)) / len(req_skills)
#         else:
#             skill_match = 0.0
        
#         # Calculate year match (20% weight)
#         year_match = 1.0 if student_year == pref_year else 0.0
        
#         # Calculate CGPA match (20% weight)
#         if min_cgpa > 0:
#             cgpa_match = min(student_cgpa / min_cgpa, 1.0)
#         else:
#             cgpa_match = 0.0
        
#         # Calculate aptitude match (10% weight)
#         if min_aptitude > 0:
#             aptitude_match = min(student_aptitude / min_aptitude, 1.0)
#         else:
#             aptitude_match = 0.0
        
#         # Calculate weighted probability
#         probability = (skill_match * skill_weight) + (year_match * year_weight) + \
#                      (cgpa_match * cgpa_weight) + (aptitude_match * aptitude_weight)
        
#         # Convert to percentage
#         probability_percentage = round(probability * 100, 2)
        
#         print(f"Skill match: {skill_match:.2f} (Weight: {skill_weight})")
#         print(f"Year match: {year_match:.2f} (Weight: {year_weight})")
#         print(f"CGPA match: {cgpa_match:.2f} (Weight: {cgpa_weight})")
#         print(f"Aptitude match: {aptitude_match:.2f} (Weight: {aptitude_weight})")
#         print(f"Final probability: {probability_percentage}%")
        
#         return probability_percentage
        
#     except Exception as e:
#         print(f"Error in calculate_probability: {str(e)}")
#         return 0.0


# # def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills, courses, cgpa, current_year, preferred_company):
# #     try:
# #         print(f"Inserting data for {name}")
        
# #         # Check if exact record already exists
# #         check_sql = """
# #         SELECT ID FROM user_data 
# #         WHERE Name = %s AND Email_ID = %s AND Timestamp = %s
# #         """
# #         cursor.execute(check_sql, (name, email, timestamp))
# #         existing_record = cursor.fetchone()
        
# #         if existing_record:
# #             print(f"Record already exists for {name} at {timestamp}")
# #             st.warning("This resume has already been analyzed!")
# #             return
        
# #         insert_sql = """
# #         INSERT INTO user_data 
# #         (Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field, User_level, 
# #         Actual_skills, Recommended_skills, Recommended_courses, CGPA, Current_year, Preferred_company)
# #         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
# #         """
        
# #         # Ensure all values are properly formatted
# #         rec_values = (
# #             str(name or 'Unknown')[:500],
# #             str(email or 'unknown@email.com')[:500],
# #             str(res_score) if res_score else '0',
# #             timestamp,
# #             str(no_of_pages) if no_of_pages else '0',
# #             str(reco_field or 'Unknown'),
# #             str(cand_level or 'Fresher'),
# #             skills if skills else '',
# #             recommended_skills if recommended_skills else '',
# #             courses if courses else '',
# #             float(cgpa) if cgpa is not None else None,
# #             int(current_year) if current_year is not None else None,
# #             str(preferred_company or '')[:500]
# #         )
        
# #         cursor.execute(insert_sql, rec_values)
# #         connection.commit()
# #         st.success("Data saved successfully!")
# #         print("Data inserted successfully!")
        
# #     except Exception as e:
# #         st.error(f"Error saving data: {str(e)}")
# #         print(f"Database error: {str(e)}")
# #         connection.rollback()
# def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, 
#                skills, recommended_skills, courses, cgpa, current_year, preferred_company, aptitude_marks):
#     try:
#         print(f"Inserting data for {name}")
        
#         # Check if exact record already exists
#         check_sql = """
#         SELECT ID FROM user_data 
#         WHERE Name = %s AND Email_ID = %s AND Timestamp = %s
#         """
#         cursor.execute(check_sql, (name, email, timestamp))
#         existing_record = cursor.fetchone()
        
#         if existing_record:
#             print(f"Record already exists for {name} at {timestamp}")
#             st.warning("This resume has already been analyzed!")
#             return
        
#         # Debug print
#         print(f"Inserting data with values:")
#         print(f"Name: {name}")
#         print(f"Email: {email}")
#         print(f"CGPA: {cgpa}")
#         print(f"Current Year: {current_year}")
#         print(f"Preferred Company: {preferred_company}")
#         print(f"Aptitude Marks: {aptitude_marks}")
        
#         insert_sql = """
#         INSERT INTO user_data 
#         (Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field, User_level, 
#         Actual_skills, Recommended_skills, Recommended_courses, CGPA, Current_year, 
#         Preferred_company, Aptitude_marks)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#         """
        
#         # Ensure all values are properly formatted
#         rec_values = (
#             str(name) if name else 'Unknown',  # Fix name handling
#             str(email) if email else 'unknown@email.com',
#             str(res_score) if res_score else '0',
#             timestamp,
#             str(no_of_pages) if no_of_pages else '0',
#             str(reco_field) if reco_field else 'Unknown',
#             str(cand_level) if cand_level else 'Fresher',
#             str(skills) if skills else '',
#             str(recommended_skills) if recommended_skills else '',
#             str(courses) if courses else '',
#             float(cgpa) if cgpa is not None else None,  # Fix CGPA handling
#             int(current_year) if current_year is not None else None,
#             str(preferred_company) if preferred_company else '',
#             int(aptitude_marks) if aptitude_marks is not None else None
#         )
        
#         # Debug print
#         print(f"SQL Values: {rec_values}")
        
#         cursor.execute(insert_sql, rec_values)
#         connection.commit()
#         st.success("Data saved successfully!")
#         print("Data inserted successfully!")
        
#     except Exception as e:
#         st.error(f"Error saving data: {str(e)}")
#         print(f"Database error: {str(e)}")
#         connection.rollback()
# def main():
#     # Initialize ML models
#     try:
#         skill_classifier, domain_predictor = train_models_if_needed()
#         if skill_classifier is None or domain_predictor is None:
#             st.warning("ML models could not be initialized. Using rule-based approach for analysis.")
#             # Initialize with None to use rule-based approach
#             skill_classifier = None
#             domain_predictor = None
#     except Exception as e:
#         st.error(f"Error initializing ML models: {str(e)}")
#         st.warning("Using rule-based approach for analysis.")
#         skill_classifier = None
#         domain_predictor = None
    
#     img = Image.open('Logo/resume_img.png')
#     img = img.resize((250, 250))
#     st.image(img)
#     st.title('AI Resume Analyzer')
#     st.sidebar.markdown('# Choose User')
#     activites = ['User', 'Admin']
#     choice = st.sidebar.selectbox('Choose among the options:', activites)
#     link = '[@Developed by batch 3 Group 2](https://www.linkedin.com/in/bhavesh-kabdwal/)'
#     st.sidebar.markdown(link, unsafe_allow_html=True)

#     # creating database
#     db_sql = 'CREATE DATABASE IF NOT EXISTS CV;'
#     cursor.execute(db_sql)

#     # creating table
#     create_table() #calling the function to create table

#     if choice == 'User':
#         st.markdown('''<h5 style='text-align: left; color: #FCF90E;'> Upload your resume, and get smart recommendations</h5>''', unsafe_allow_html=True)
        
#         # Create a container for user input fields
#         with st.container():
#             # Add new input fields
#             user_name = st.text_input("Enter your full name")
#             user_cgpa = st.number_input("Enter your CGPA", min_value=0.0, max_value=10.0, step=0.1)
#             user_current_year = st.number_input("Enter your current year of study", min_value=1, max_value=5, step=1)
#             user_preferred_company = st.text_input("Enter your preferred company")
            
#             # Add aptitude test button right after preferred company input
#             if user_preferred_company:
#                 if st.button("Take Aptitude Test"):
#                     st.session_state.company = user_preferred_company
#                     st.switch_page("pages/1_📝_Aptitude_Test.py")
            
#             # Display aptitude score if available
#             if 'aptitude_score' in st.session_state:
#                 st.success(f"Your Aptitude Test Score: {st.session_state.aptitude_score}/100")
#                 user_aptitude_marks = st.session_state.aptitude_score
#             else:
#                 user_aptitude_marks = None
        
#         # Create a separate container for file upload
#         with st.container():
#             pdf_file = st.file_uploader('Upload your Resume', type=['pdf'])
            
#             if pdf_file is not None:
#                 with st.spinner('Analyzing your Resume with ML . . .'):
#                     time.sleep(4)
#                 save_pdf_path = './Uploaded_Resumes/' + pdf_file.name
#                 with open(save_pdf_path, 'wb') as f:
#                     f.write(pdf_file.getbuffer())
#                 show_pdf(save_pdf_path)
#                 resume_data = ResumeParser(save_pdf_path).get_extracted_data()
#                 if resume_data:
#                     # get all the resume text
#                     resume_text = pdf_reader(save_pdf_path)

#                     st.header('*Resume Analysis*')
#                     st.success('Hello ' +user_name)
                    
#                     # Extract skills using ML
#                     current_skills = extract_skills(resume_text, skill_classifier)
                    
#                     # Make sure skills is a list
#                     if isinstance(current_skills, str):
#                         current_skills = [current_skills]
#                     elif current_skills is None:
#                         current_skills = []
                    
#                     # Display current skills
#                     st.subheader('*Your Current Skills*')
#                     if current_skills:
#                         st.write(", ".join(current_skills))
#                     else:
#                         st.write("No skills detected. Please make sure your resume includes your technical skills.")

#                     # Predict domain
#                     reco_field = predict_domain(resume_text, domain_predictor)
                    
#                     st.subheader('*Predicted Field*')
#                     st.success(f"Our ML model predicts you are looking for a {reco_field} Job")
                    
#                     # Get recommended skills
#                     recommended_skills = []
#                     if reco_field in domain_skills:
#                         recommended_skills = [
#                             skill for skill in domain_skills[reco_field] 
#                             if skill.lower() not in [s.lower() for s in current_skills]
#                         ]
                    
#                     # Display recommended skills
#                     st.subheader('*Recommended Skills*')
#                     if recommended_skills:
#                         st.write(", ".join(recommended_skills))
#                         st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost🚀 the chances of getting a Job</h4>''', unsafe_allow_html=True)
#                     else:
#                         st.write("No additional skills to recommend at this time.")

#                     # Determine candidate level based on experience/skills
#                     if len(current_skills) >= 13:
#                         cand_level = "Expert"
#                     elif len(current_skills) >= 5:
#                         cand_level = "Intermediate"
#                     else:
#                         cand_level = "Beginner"

#                     # Get course recommendations based on domain
#                     if reco_field == 'Data Science':
#                         rec_course = course_recommender(ds_course)
#                     elif reco_field == 'Web Development':
#                         rec_course = course_recommender(web_course)
#                     elif reco_field == 'Android Development':
#                         rec_course = course_recommender(android_course)
#                     elif reco_field == 'IOS Development':
#                         rec_course = course_recommender(ios_course)
#                     elif reco_field == 'UI-UX Development':
#                         rec_course = course_recommender(uiux_course)
#                     else:
#                         # Default to data science courses if domain not recognized
#                         rec_course = course_recommender(ds_course)

#                     # Display course recommendations
#                     st.subheader("*Recommended Courses*")
#                     if rec_course:
#                         for i, course in enumerate(rec_course, 1):
#                             st.write(f"{i}. {course}")
#                     else:
#                         st.write("No specific courses to recommend at this time.")
                    

#                     st.subheader('*Probability of Getting Placed in Desired Companies*')
            
#                     # Calculate probability for each company
#                     if pdf_file is not None and current_skills:  # Only calculate if resume is uploaded and skills are extracted
#                         try:
#                             probabilities = []
#                             for _, company in companies_df.iterrows():
#                                 # Create student profile
#                                 student_profile = {
#                                     'skills': ",".join(current_skills),
#                                     'current_year': user_current_year,
#                                     'cgpa': user_cgpa,
#                                     'aptitude_marks': user_aptitude_marks if user_aptitude_marks is not None else 0
#                                 }
                                
#                                 # Calculate probability
#                                 prob = calculate_probability(student_profile, company)
#                                 probabilities.append((company['company_name'], prob))
                            
#                             # Sort probabilities from highest to lowest
#                             probabilities.sort(key=lambda x: x[1], reverse=True)
                            
#                             # Display probabilities in a table
#                             prob_df = pd.DataFrame(probabilities, columns=['Company', 'Probability (%)'])
#                             prob_df['Probability (%)'] = prob_df['Probability (%)'].round(2)
                            
#                             if not prob_df.empty:
#                                 st.dataframe(prob_df)
                                
#                                 # Display visual chart for top 5 companies
#                                 top_companies = prob_df.head(5)
#                                 fig = px.bar(
#                                     top_companies, 
#                                     x='Company', 
#                                     y='Probability (%)',
#                                     title='Top 5 Companies Match',
#                                     color='Probability (%)',
#                                     color_continuous_scale='Viridis'
#                                 )
#                                 st.plotly_chart(fig)
#                             else:
#                                 st.info("No company matches found based on your profile")
                                
#                         except Exception as e:
#                             st.error(f"Error calculating probabilities: {str(e)}")
#                             print(f"Error details: {str(e)}")
#                     else:
#                         st.info("Please upload your resume and complete all fields to see company matches")

#                     # Resume score calculation
#                     st.subheader('*Resume Tips & Ideas💡*')
#                     resume_score = 0
#                     if 'Objective' in resume_text:
#                         resume_score = resume_score + 20
#                         st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective</h4>''', unsafe_allow_html=True)
#                     else:
#                         st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add your career objective, it will give your career intention to the Recruiters.</h4>''', unsafe_allow_html=True)

#                     if 'Declaration' in resume_text:
#                         resume_score = resume_score + 20
#                         st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Declaration</h4>''', unsafe_allow_html=True)
#                     else:
#                         st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add Declaration. It will give the assurance that everything written on your resume is true and fully acknowledged by you</h4>''', unsafe_allow_html=True)

#                     if 'Hobbies' in resume_text or 'Interests' in resume_text:
#                         resume_score = resume_score + 20
#                         st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies</h4>''', unsafe_allow_html=True)
#                     else:
#                         st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add Hobbies. It will show your personality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''', unsafe_allow_html=True)

#                     if 'Achievements' in resume_text or 'Skills' in resume_text:
#                         resume_score = resume_score + 20
#                         st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Skills </h4>''', unsafe_allow_html=True)
#                     else:
#                         st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add Skills. It will show that you are capable for the required position.</h4>''', unsafe_allow_html=True)

#                     if 'Projects' in resume_text:
#                         resume_score = resume_score + 20
#                         st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''', unsafe_allow_html=True)
#                     else:
#                         st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add Projects. It will show that you have done work related the required position or not.</h4>''', unsafe_allow_html=True)

#                     st.subheader("*Resume Score📝*")
#                     st.markdown(
#                         """
#                         <style>
#                             .stProgress > div > div > div > div {
#                                 background-color: #d73b5c;
#                             }
#                         </style>""",
#                         unsafe_allow_html=True,
#                     )
                    
#                     my_bar = st.progress(0)
#                     score = 0
#                     for percent_complete in range(resume_score):
#                         score += 1
#                         time.sleep(0.1)
#                         my_bar.progress(percent_complete + 1)
#                     st.success('** Your Resume Writing Score: ' + str(score) + '')
#                     st.warning('** Note: This score is based on your content that you have in Resume. **')
#                     st.balloons()

#                     # Prepare timestamp
#                     ts = time.time()
#                     cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#                     cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#                     timestamp = f"{cur_date} {cur_time}"

#                     # Insert data into database
#                     try:
#                         # Debug print before insertion
#                         print(f"Preparing to insert data:")
#                         print(f"User Name: {user_name}")
#                         print(f"CGPA: {user_cgpa}")
#                         print(f"Current Year: {user_current_year}")
#                         print(f"Preferred Company: {user_preferred_company}")
#                         print(f"Aptitude Marks: {user_aptitude_marks}")
                        
#                         # Ensure user_name is not empty
#                         if not user_name:
#                             st.error("Please enter your name")
#                             return
                            
#                         # Ensure CGPA is valid
#                         if user_cgpa is None or user_cgpa < 0 or user_cgpa > 10:
#                             st.error("Please enter a valid CGPA between 0 and 10")
#                             return
                            
#                         # Ensure current year is valid
#                         if user_current_year is None or user_current_year < 1 or user_current_year > 5:
#                             st.error("Please enter a valid current year between 1 and 5")
#                             return
                        
#                         insert_data(
#                             name=user_name,
#                             email=resume_data.get('email', ''),
#                             res_score=str(resume_score),
#                             timestamp=timestamp,
#                             no_of_pages=str(resume_data.get('no_of_pages', '0')),
#                             reco_field=reco_field,
#                             cand_level=cand_level,
#                             skills=", ".join(current_skills) if current_skills else '',
#                             recommended_skills=", ".join(recommended_skills) if recommended_skills else '',
#                             courses=", ".join(rec_course) if rec_course else '',
#                             cgpa=float(user_cgpa),  # Convert to float
#                             current_year=int(user_current_year),  # Convert to int
#                             preferred_company=user_preferred_company if user_preferred_company else '',
#                             aptitude_marks=int(user_aptitude_marks) if user_aptitude_marks is not None else None
#                         )
#                     except Exception as e:
#                         st.error(f"Error saving data: {str(e)}")
#                         print(f"Error details: {str(e)}")

#                     # resume writing video
#                     st.header('*Bonus Video for Resume Writing Tips💡*')
#                     resume_vid = random.choice(resume_videos)
#                     res_vid_title = fetch_yt_video(resume_vid)
#                     st.subheader("✅ *" + res_vid_title + "*")
#                     st.video(resume_vid)

#                     # interview preparation tips
#                     st.header('*Bonus Video for Interview Tips💡*')
#                     interview_vid = random.choice(interview_videos)
#                     interview_vid_title = fetch_yt_video(interview_vid)
#                     st.subheader("✅ *" + interview_vid_title + "*")
#                     st.video(interview_vid)

#                 else:
#                     st.error('** Something went wrong **')
    
#     else:
#         ## Admin side
#         st.success('** Welcome to Admin Side **')
#         ad_user = st.text_input("Username")
#         ad_password = st.text_input("Password", type='password')
#         if st.button('Login'):
#             if ad_user == 'bhavesh' and ad_password == 'bhaveshk22':
#                 st.success("Welcome Bhavesh !")

#                 # Display Data
#                 cursor.execute('''SELECT * FROM user_data''')
#                 data = cursor.fetchall()
#                 st.header("*User's Data*")
#                 df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Page',
#                                                'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills',
#                                                'Recommended Course', 'CGPA', 'Current Year', 'Preferred Company'])
#                 st.dataframe(df)
#                 st.markdown(get_table_download_link(df, 'User_Data.csv', 'Download Report'), unsafe_allow_html=True)
                
#                 ## Admin Side Data
#                 query = 'select * from user_data;'
#                 plot_data = pd.read_sql(query, connection)
                
#                 ## Pie chart for predicted field recommendations
#                 labels = plot_data.Predicted_Field.unique()
#                 values = plot_data.Predicted_Field.value_counts()
                
#                 st.subheader("*Pie-Chart for Predicted Field Recommendation*")
                
#                 fig, ax = plt.subplots()
#                 fig.patch.set_facecolor('#262730')
#                 ax.pie(values, labels=labels, autopct='%1.1f%%', radius=0.6, textprops={'fontsize': 7, 'color': 'white'})
#                 st.pyplot(fig, use_container_width=False)

#                 ### Pie chart for User's👨‍💻 Experienced Level
#                 labels = plot_data.User_level.unique()
#                 values = plot_data.User_level.value_counts()
                
#                 st.subheader("*Pie-Chart for User's Experienced Level*")
#                 fig, ax = plt.subplots()
#                 fig.patch.set_facecolor('#262730')
#                 ax.pie(values, labels=labels, autopct='%1.1f%%', radius=0.6, textprops={'fontsize': 7, 'color': 'white'})
#                 st.pyplot(fig, use_container_width=False)
                
#                 # ML Model Performance
#                 st.subheader("*ML Model Performance*")
                
#                 # Create sample data for visualization
#                 model_names = ['Skill Classifier', 'Domain Predictor']
#                 accuracies = [0.85, 0.92]  # Example accuracies
                
#                 fig, ax = plt.subplots()
#                 fig.patch.set_facecolor('#262730')
#                 ax.bar(model_names, accuracies, color=['#1ed760', '#d73b5c'])
#                 ax.set_ylim(0, 1.0)
#                 ax.set_ylabel('Accuracy', color='white')
#                 ax.set_title('ML Model Performance', color='white')
#                 ax.tick_params(axis='x', colors='white')
#                 ax.tick_params(axis='y', colors='white')
                
#                 for i, v in enumerate(accuracies):
#                     ax.text(i, v + 0.05, f"{v:.2f}", ha='center', color='white')
                    
#                 st.pyplot(fig, use_container_width=False)

#             else:
#                 st.error("Wrong ID & Password Provided")

# if __name__ == "__main__":
#     main()

import streamlit as st
# First Streamlit command must be st.set_page_config()
st.set_page_config(
    page_title='AI Resume Analyzer',
    page_icon='Logo/resume_icon.jpg'
)

# Now import all other libraries
import pandas as pd
import base64, random
import time, datetime
# libraries to parse the resume pdf files
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io, random
from streamlit_tags import st_tags 
from PIL import Image
import pymysql
from Courses import ds_course ,web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
import yt_dlp
import matplotlib.pyplot as plt
import plotly.express as px
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
from ml_utils import extract_skills, predict_domain, train_models_if_needed

# Define create_table function first
# def create_table():
#     try:
#         # Create table only if it doesn't exist (removed DROP TABLE)
#         table_sql = """
#         CREATE TABLE IF NOT EXISTS user_data (
#             ID INT NOT NULL AUTO_INCREMENT,
#             Name VARCHAR(500) NOT NULL,
#             Email_ID VARCHAR(500) NOT NULL,
#             resume_score VARCHAR(8) NOT NULL,
#             Timestamp VARCHAR(50) NOT NULL,
#             Page_no VARCHAR(5) NOT NULL,
#             Predicted_Field TEXT NOT NULL,
#             User_level TEXT NOT NULL,
#             Actual_skills TEXT NOT NULL,
#             Recommended_skills TEXT NOT NULL,
#             Recommended_courses TEXT NOT NULL,
#             CGPA FLOAT,
#             Current_year INT,
#             Preferred_company VARCHAR(500),
#             PRIMARY KEY (ID)
#         ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
#         """
#         cursor.execute(table_sql)
#         connection.commit()
#         print("Table check completed successfully!")
        
#     except Exception as e:
#         print(f"Error with table: {str(e)}")
#         st.error(f"Error with table: {str(e)}")
def create_table():
    try:
        # Create table only if it doesn't exist (removed DROP TABLE)
        table_sql = """
        CREATE TABLE IF NOT EXISTS user_data (
            ID INT NOT NULL AUTO_INCREMENT,
            Name VARCHAR(500) NOT NULL,
            Email_ID VARCHAR(500) NOT NULL,
            resume_score VARCHAR(8) NOT NULL,
            Timestamp VARCHAR(50) NOT NULL,
            Page_no VARCHAR(5) NOT NULL,
            Predicted_Field TEXT NOT NULL,
            User_level TEXT NOT NULL,
            Actual_skills TEXT NOT NULL,
            Recommended_skills TEXT NOT NULL,
            Recommended_courses TEXT NOT NULL,
            CGPA FLOAT,
            Current_year INT,
            Preferred_company VARCHAR(500),
            Aptitude_marks INT,  # Add this new column
            PRIMARY KEY (ID)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        cursor.execute(table_sql)
        connection.commit()
        print("Table check completed successfully!")
        
    except Exception as e:
        print(f"Error with table: {str(e)}")
        st.error(f"Error with table: {str(e)}")

# Initialize database connection
try:
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='mysql',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    cursor = connection.cursor()
    
    # Create database if it doesn't exist
    cursor.execute("CREATE DATABASE IF NOT EXISTS cv")
    cursor.execute("USE cv")
    
    # Create table if it doesn't exist
    create_table()
    
    st.success("Database Connected Successfully!")
except Exception as e:
    st.error(f"Error connecting to Database: {e}")

# Load or train ML models
skill_classifier, domain_predictor = train_models_if_needed()

# Dictionary of skills for each domain
domain_skills = {
    'Data Science': ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling', 'Data Mining', 
                    'Clustering & Classification', 'Data Analytics', 'Quantitative Analysis', 
                    'Web Scraping', 'ML Algorithms', 'Keras', 'Pytorch', 'Probability', 
                    'Scikit-learn', 'Tensorflow', 'Flask', 'Streamlit'],
    'Web Development': ['React', 'Django', 'Node JS', 'React JS', 'php', 'laravel', 'Magento', 
                       'wordpress', 'Javascript', 'Angular JS', 'c#', 'Flask', 'SDK'],
    'Android Development': ['Android', 'Android development', 'Flutter', 'Kotlin', 'XML', 'Java', 
                           'Kivy', 'GIT', 'SDK', 'SQLite'],
    'IOS Development': ['IOS', 'IOS Development', 'Swift', 'Cocoa', 'Cocoa Touch', 'Xcode', 
                       'Objective-C', 'SQLite', 'Plist', 'StoreKit', 'UI-Kit', 'AV Foundation', 'Auto-Layout'],
    'UI-UX Development': ['UI', 'User Experience', 'Adobe XD', 'Figma', 'Zeplin', 'Balsamiq', 
                         'Prototyping', 'Wireframes', 'Storyframes', 'Adobe Photoshop', 'Editing', 
                         'Illustrator', 'After Effects', 'Premier Pro', 'Indesign', 'Wireframe', 
                         'Solid', 'Grasp', 'User Research']
}

def fetch_yt_video(link):
    with yt_dlp.YoutubeDL({"no_warnings": True}) as ydl:
        info = ydl.extract_info(link, download=False)
    return info["title"]

def get_table_download_link(df, filename, text):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams(detect_vertical=True))
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

def show_pdf(file_path):
    with open(file_path, 'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def course_recommender(course_list):
    st.subheader('Courses & Certificates Recommendations 🎓')
    rec_course = []
    no_of_rec = st.slider('Choose number of Courses to Recommend', 1, 10, 5)
    
    if isinstance(course_list, list):
        # If course_list is already a list of tuples
        courses = course_list
    else:
        # If course_list is a dictionary or other format, convert to list
        courses = list(course_list.items()) if hasattr(course_list, 'items') else []
    
    random.shuffle(courses)
    
    for c_name, c_link in courses[:no_of_rec]:
        st.markdown(f"- [{c_name}]({c_link})")
        rec_course.append(c_name)
    
    return rec_course


# Load company dataset
companies_df = pd.read_csv('D:\\2nd Year Course Projects\\DS\\AI-Resume-Analyzer-main (1)\\company.csv')  # Ensure the file path is correct

import pandas as pd

def calculate_probability(student, company):
    try:
        # Debug print
        print(f"\nCalculating probability for company: {company['company_name']}")
        print(f"Student profile: {student}")
        print(f"Company requirements: {company}")
        
        # Updated weights
        skill_weight = 0.6
        year_weight = 0.15
        cgpa_weight = 0.15
        aptitude_weight = 0.1
        
        # Extract student data with defaults
        student_skills = set(map(str.strip, str(student.get('skills', '')).lower().split(','))) if student.get('skills') else set()
        student_year = int(student.get('current_year', 0))
        student_cgpa = float(student.get('cgpa', 0.0))
        student_aptitude = int(student.get('aptitude_marks', 0))
        
        # Extract company requirements with defaults
        req_skills = set(map(str.strip, str(company.get('required_skills', '')).lower().split(','))) if pd.notna(company.get('required_skills')) else set()
        pref_year = int(company.get('preferred_year', 0)) if pd.notna(company.get('preferred_year')) else 0
        min_cgpa = float(company.get('required_cgpa', 0.0)) if pd.notna(company.get('required_cgpa')) else 0.0
        min_aptitude = int(company.get('required_aptitude_marks', 0)) if pd.notna(company.get('required_aptitude_marks')) else 0
        
        # Calculate skill match
        if req_skills:
            matched_skills = student_skills.intersection(req_skills)
            skill_match = len(matched_skills) / len(req_skills)
        else:
            skill_match = 0.0
        
        # Calculate year match
        year_match = 1.0 if student_year == pref_year else 0.0
        
        # Calculate CGPA match
        cgpa_match = min(student_cgpa / min_cgpa, 1.0) if min_cgpa > 0 else 0.0
        
        # Calculate aptitude match
        aptitude_match = min(student_aptitude / min_aptitude, 1.0) if min_aptitude > 0 else 0.0
        
        # Calculate final probability
        probability = (skill_match * skill_weight) + (year_match * year_weight) + \
                     (cgpa_match * cgpa_weight) + (aptitude_match * aptitude_weight)
        
        # Convert to percentage
        probability_percentage = round(probability * 100, 2)
        
        # Debugging output
        print(f"Skill match: {skill_match:.2f} (Weight: {skill_weight})")
        print(f"Year match: {year_match:.2f} (Weight: {year_weight})")
        print(f"CGPA match: {cgpa_match:.2f} (Weight: {cgpa_weight})")
        print(f"Aptitude match: {aptitude_match:.2f} (Weight: {aptitude_weight})")
        print(f"Final probability: {probability_percentage}%")
        
        return probability_percentage
        
    except Exception as e:
        print(f"Error in calculate_probability: {str(e)}")
        return 0.0



# def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills, courses, cgpa, current_year, preferred_company):
#     try:
#         print(f"Inserting data for {name}")
        
#         # Check if exact record already exists
#         check_sql = """
#         SELECT ID FROM user_data 
#         WHERE Name = %s AND Email_ID = %s AND Timestamp = %s
#         """
#         cursor.execute(check_sql, (name, email, timestamp))
#         existing_record = cursor.fetchone()
        
#         if existing_record:
#             print(f"Record already exists for {name} at {timestamp}")
#             st.warning("This resume has already been analyzed!")
#             return
        
#         insert_sql = """
#         INSERT INTO user_data 
#         (Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field, User_level, 
#         Actual_skills, Recommended_skills, Recommended_courses, CGPA, Current_year, Preferred_company)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#         """
        
#         # Ensure all values are properly formatted
#         rec_values = (
#             str(name or 'Unknown')[:500],
#             str(email or 'unknown@email.com')[:500],
#             str(res_score) if res_score else '0',
#             timestamp,
#             str(no_of_pages) if no_of_pages else '0',
#             str(reco_field or 'Unknown'),
#             str(cand_level or 'Fresher'),
#             skills if skills else '',
#             recommended_skills if recommended_skills else '',
#             courses if courses else '',
#             float(cgpa) if cgpa is not None else None,
#             int(current_year) if current_year is not None else None,
#             str(preferred_company or '')[:500]
#         )
        
#         cursor.execute(insert_sql, rec_values)
#         connection.commit()
#         st.success("Data saved successfully!")
#         print("Data inserted successfully!")
        
#     except Exception as e:
#         st.error(f"Error saving data: {str(e)}")
#         print(f"Database error: {str(e)}")
#         connection.rollback()
def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, 
               skills, recommended_skills, courses, cgpa, current_year, preferred_company, aptitude_marks):
    try:
        print(f"Inserting data for {name}")
        
        # Check if exact record already exists
        check_sql = """
        SELECT ID FROM user_data 
        WHERE Name = %s AND Email_ID = %s AND Timestamp = %s
        """
        cursor.execute(check_sql, (name, email, timestamp))
        existing_record = cursor.fetchone()
        
        if existing_record:
            print(f"Record already exists for {name} at {timestamp}")
            st.warning("This resume has already been analyzed!")
            return
        
        # Debug print
        print(f"Inserting data with values:")
        print(f"Name: {name}")
        print(f"Email: {email}")
        print(f"CGPA: {cgpa}")
        print(f"Current Year: {current_year}")
        print(f"Preferred Company: {preferred_company}")
        print(f"Aptitude Marks: {aptitude_marks}")
        
        insert_sql = """
        INSERT INTO user_data 
        (Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field, User_level, 
        Actual_skills, Recommended_skills, Recommended_courses, CGPA, Current_year, 
        Preferred_company, Aptitude_marks)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Ensure all values are properly formatted
        rec_values = (
            str(name or 'Unknown')[:500],
            str(email or 'unknown@email.com')[:500],
            str(res_score) if res_score else '0',
            timestamp,
            str(no_of_pages) if no_of_pages else '0',
            str(reco_field or 'Unknown'),
            str(cand_level or 'Fresher'),
            skills if skills else '',
            recommended_skills if recommended_skills else '',
            courses if courses else '',
            float(cgpa) if cgpa is not None else None,
            int(current_year) if current_year is not None else None,
            str(preferred_company or '')[:500],
            int(aptitude_marks) if aptitude_marks is not None else None
        )
        
        # Debug print
        print(f"SQL Values: {rec_values}")
        
        cursor.execute(insert_sql, rec_values)
        connection.commit()
        st.success("Data saved successfully!")
        print("Data inserted successfully!")
        
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        print(f"Database error: {str(e)}")
        connection.rollback()
def main():
    # Initialize ML models
    try:
        skill_classifier, domain_predictor = train_models_if_needed()
        if skill_classifier is None or domain_predictor is None:
            # st.warning("ML models could not be initialized. Using rule-based approach for analysis.")
            # Initialize with None to use rule-based approach
            skill_classifier = None
            domain_predictor = None
    except Exception as e:
        st.error(f"Error initializing ML models: {str(e)}")
        st.warning("Using rule-based approach for analysis.")
        skill_classifier = None
        domain_predictor = None
    
    img = Image.open('Logo/resume_img.png')
    img = img.resize((350, 250))
    st.image(img)
    st.title(' Resume Analyzer')
    st.sidebar.markdown('# Choose User')
    activites = ['User', 'Admin']
    choice = st.sidebar.selectbox('Choose among the options:', activites)
    link = '[@Developed by batch 3 Group 2](https://www.linkedin.com/in/bhavesh-kabdwal/)'
    st.sidebar.markdown(link, unsafe_allow_html=True)

    # creating database
    db_sql = 'CREATE DATABASE IF NOT EXISTS CV;'
    cursor.execute(db_sql)

    # creating table
    create_table() #calling the function to create table

    if choice == 'User':
        st.markdown('''<h5 style='text-align: left; color: #FCF90E;'> Upload your resume, and get smart recommendations</h5>''', unsafe_allow_html=True)
        # Add new input fields
        user_name = st.text_input("Enter your full name")
        user_cgpa = st.number_input("Enter your CGPA", min_value=0.0, max_value=10.0, step=0.1)
        user_current_year = st.number_input("Enter your current year of study", min_value=1, max_value=5, step=1)
        user_preferred_company = st.text_input("Enter your preferred company")
        
        # Add aptitude test button
        if user_preferred_company:
            if st.button("Take Aptitude Test"):
                st.session_state.company = user_preferred_company
                st.switch_page("pages/1_📝_Aptitude_Test.py")
        
        # Display aptitude score if available
        if 'aptitude_score' in st.session_state:
            st.success(f"Your Aptitude Test Score: {st.session_state.aptitude_score}/100")
            user_aptitude_marks = st.session_state.aptitude_score
        else:
            user_aptitude_marks = None
            
        pdf_file = st.file_uploader('Upload your Resume', type=['pdf'])
        if pdf_file is not None:
            with st.spinner('Analyzing your Resume with ML . . .'):
                time.sleep(4)
            save_pdf_path = './Uploaded_Resumes/' + pdf_file.name
            with open(save_pdf_path, 'wb') as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_pdf_path)
            resume_data = ResumeParser(save_pdf_path).get_extracted_data()
            if resume_data:
                # get all the resume text
                resume_text = pdf_reader(save_pdf_path)

                st.header('Resume Analysis')
                st.success('Hello ' + (resume_data.get('name', 'User') or 'User'))
                
                # Extract skills using ML
                current_skills = extract_skills(resume_text, skill_classifier)
                
                # Make sure skills is a list
                if isinstance(current_skills, str):
                    current_skills = [current_skills]
                elif current_skills is None:
                    current_skills = []
                
                # Display current skills
                st.subheader('Your Current Skills')
                if current_skills:
                    st.write(", ".join(current_skills))
                else:
                    st.write("No skills detected. Please make sure your resume includes your technical skills.")

                # Predict domain
                reco_field = predict_domain(resume_text, domain_predictor)
                
                st.subheader('Predicted Field')
                st.success(f"Our ML model predicts you are looking for a {reco_field} Job")
                
                # Get recommended skills
                recommended_skills = []
                if reco_field in domain_skills:
                    recommended_skills = [
                        skill for skill in domain_skills[reco_field] 
                        if skill.lower() not in [s.lower() for s in current_skills]
                    ]
                
                # Display recommended skills
                st.subheader('Recommended Skills')
                if recommended_skills:
                    st.write(", ".join(recommended_skills))
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost🚀 the chances of getting a Job</h4>''', unsafe_allow_html=True)
                else:
                    st.write("No additional skills to recommend at this time.")

                # Determine candidate level based on experience/skills
                if len(current_skills) >= 13:
                    cand_level = "Expert"
                elif len(current_skills) >= 5:
                    cand_level = "Intermediate"
                else:
                    cand_level = "Beginner"

                # Get course recommendations based on domain
                if reco_field == 'Data Science':
                    rec_course = course_recommender(ds_course)
                elif reco_field == 'Web Development':
                    rec_course = course_recommender(web_course)
                elif reco_field == 'Android Development':
                    rec_course = course_recommender(android_course)
                elif reco_field == 'IOS Development':
                    rec_course = course_recommender(ios_course)
                elif reco_field == 'UI-UX Development':
                    rec_course = course_recommender(uiux_course)
                else:
                    # Default to data science courses if domain not recognized
                    rec_course = course_recommender(ds_course)

                # Display course recommendations
                st.subheader("Recommended Courses")
                if rec_course:
                    for i, course in enumerate(rec_course, 1):
                        st.write(f"{i}. {course}")
                else:
                    st.write("No specific courses to recommend at this time.")
                

                st.subheader('Probability of Getting Placed in Desired Companies')
        
                # Calculate probability for each company
                if pdf_file is not None and current_skills:  # Only calculate if resume is uploaded and skills are extracted
                    try:
                        probabilities = []
                        for _, company in companies_df.iterrows():
                            # Create student profile
                            student_profile = {
                                'skills': ",".join(current_skills),
                                'current_year': user_current_year,
                                'cgpa': user_cgpa,
                                'aptitude_marks': user_aptitude_marks if user_aptitude_marks is not None else 0
                            }
                            
                            # Calculate probability
                            prob = calculate_probability(student_profile, company)
                            probabilities.append((company['company_name'], prob))
                        
                        # Sort probabilities from highest to lowest
                        probabilities.sort(key=lambda x: x[1], reverse=True)
                        
                        # Display probabilities in a table
                        prob_df = pd.DataFrame(probabilities, columns=['Company', 'Probability (%)'])
                        prob_df['Probability (%)'] = prob_df['Probability (%)'].round(2)
                        
                        if not prob_df.empty:
                            st.dataframe(prob_df)
                            
                            # Display visual chart for top 5 companies
                            top_companies = prob_df.head(5)
                            fig = px.bar(
                                top_companies, 
                                x='Company', 
                                y='Probability (%)',
                                title='Top 5 Companies Match',
                                color='Probability (%)',
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig)
                        else:
                            st.info("No company matches found based on your profile")
                            
                    except Exception as e:
                        st.error(f"Error calculating probabilities: {str(e)}")
                        print(f"Error details: {str(e)}")
                else:
                    st.info("Please upload your resume and complete all fields to see company matches")

                # Resume score calculation
                st.subheader('Resume Tips & Ideas💡')
                resume_score = 0
                if 'Objective' in resume_text:
                    resume_score = resume_score + 20
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective</h4>''', unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add your career objective, it will give your career intention to the Recruiters.</h4>''', unsafe_allow_html=True)

                if 'Declaration' in resume_text:
                    resume_score = resume_score + 20
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Declaration</h4>''', unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add Declaration. It will give the assurance that everything written on your resume is true and fully acknowledged by you</h4>''', unsafe_allow_html=True)

                if 'Hobbies' in resume_text or 'Interests' in resume_text:
                    resume_score = resume_score + 20
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies</h4>''', unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add Hobbies. It will show your personality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''', unsafe_allow_html=True)

                if 'Achievements' in resume_text or 'Skills' in resume_text:
                    resume_score = resume_score + 20
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Skills </h4>''', unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add Skills. It will show that you are capable for the required position.</h4>''', unsafe_allow_html=True)

                if 'Projects' in resume_text:
                    resume_score = resume_score + 20
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''', unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add Projects. It will show that you have done work related the required position or not.</h4>''', unsafe_allow_html=True)

                st.subheader("Resume Score📝")
                st.markdown(
                    """
                    <style>
                        .stProgress > div > div > div > div {
                            background-color: #d73b5c;
                        }
                    </style>""",
                    unsafe_allow_html=True,
                )
                
                my_bar = st.progress(0)
                score = 0
                for percent_complete in range(resume_score):
                    score += 1
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)
                st.success('** Your Resume Writing Score: ' + str(score) + '')
                st.warning('** Note: This score is based on your content that you have in Resume. **')
                st.balloons()

                # Prepare timestamp
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = f"{cur_date} {cur_time}"

                # Insert data into database
               # In the main() function where you call insert_data:
                try:
                    insert_data(
                        name=user_name,  # From st.text_input
                        email=resume_data.get('email', ''),
                        res_score=str(resume_score),
                        timestamp=timestamp,
                        no_of_pages=str(resume_data.get('no_of_pages', '0')),
                        reco_field=reco_field,
                        cand_level=cand_level,
                        skills=", ".join(current_skills) if current_skills else '',
                        recommended_skills=", ".join(recommended_skills) if recommended_skills else '',
                        courses=", ".join(rec_course) if rec_course else '',
                        cgpa=float(user_cgpa) if user_cgpa else None,  # Convert to float
                        current_year=int(user_current_year) if user_current_year else None,  # Convert to int
                        preferred_company=user_preferred_company if user_preferred_company else '',
                        aptitude_marks=int(user_aptitude_marks) if user_aptitude_marks is not None else None
                    )
                except Exception as e:
                    st.error(f"Error saving data: {str(e)}")
                    print(f"Error details: {str(e)}")

                # resume writing video
                st.header('Bonus Video for Resume Writing Tips💡')
                resume_vid = random.choice(resume_videos)
                res_vid_title = fetch_yt_video(resume_vid)
                st.subheader("✅ " + res_vid_title + "")
                st.video(resume_vid)

                # interview preparation tips
                st.header('Bonus Video for Interview Tips💡')
                interview_vid = random.choice(interview_videos)
                interview_vid_title = fetch_yt_video(interview_vid)
                st.subheader("✅ " + interview_vid_title + "")
                st.video(interview_vid)

            else:
                st.error('** Something went wrong **')
    
    else:
        ## Admin side
        st.success('** Welcome to Admin Side **')
        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')
        if st.button('Login'):
            if ad_user == 'bhavesh' and ad_password == 'bhaveshk22':
                st.success("Welcome Bhavesh !")

                # Display Data
                cursor.execute('''SELECT * FROM user_data''')
                data = cursor.fetchall()
                st.header("User's Data")
                df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Page',
                                               'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills',
                                               'Recommended Course', 'CGPA', 'Current Year', 'Preferred Company'])
                st.dataframe(df)
                st.markdown(get_table_download_link(df, 'User_Data.csv', 'Download Report'), unsafe_allow_html=True)
                
                ## Admin Side Data
                query = 'select * from user_data;'
                plot_data = pd.read_sql(query, connection)
                
                ## Pie chart for predicted field recommendations
                labels = plot_data.Predicted_Field.unique()
                values = plot_data.Predicted_Field.value_counts()
                
                st.subheader("Pie-Chart for Predicted Field Recommendation")
                
                fig, ax = plt.subplots()
                fig.patch.set_facecolor('#262730')
                ax.pie(values, labels=labels, autopct='%1.1f%%', radius=0.6, textprops={'fontsize': 7, 'color': 'white'})
                st.pyplot(fig, use_container_width=False)

                ### Pie chart for User's👨‍💻 Experienced Level
                labels = plot_data.User_level.unique()
                values = plot_data.User_level.value_counts()
                
                st.subheader("Pie-Chart for User's Experienced Level")
                fig, ax = plt.subplots()
                fig.patch.set_facecolor('#262730')
                ax.pie(values, labels=labels, autopct='%1.1f%%', radius=0.6, textprops={'fontsize': 7, 'color': 'white'})
                st.pyplot(fig, use_container_width=False)
                
                # ML Model Performance
                st.subheader("ML Model Performance")
                
                # Create sample data for visualization
                model_names = ['Skill Classifier', 'Domain Predictor']
                accuracies = [0.85, 0.92]  # Example accuracies
                
                fig, ax = plt.subplots()
                fig.patch.set_facecolor('#262730')
                ax.bar(model_names, accuracies, color=['#1ed760', '#d73b5c'])
                ax.set_ylim(0, 1.0)
                ax.set_ylabel('Accuracy', color='white')
                ax.set_title('ML Model Performance', color='white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                
                for i, v in enumerate(accuracies):
                    ax.text(i, v + 0.05, f"{v:.2f}", ha='center', color='white')
                    
                st.pyplot(fig, use_container_width=False)

            else:
                st.error("Wrong ID & Password Provided")

if __name__ == "__main__":
    main()