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
    st.subheader('*Courses & Certificates Recommendations 🎓*')
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
companies_df = pd.read_csv('C:\\Users\\Asus\\Documents\\company.csv')  # Ensure the file path is correct
def calculate_probability(student, company):
    # Extract required fields
    required_skills = set(company['required_skills'].split(','))
    preferred_year = company['preferred_year']
    required_cgpa = company['required_cgpa'] 
    required_aptitude_marks = company['required_aptitude_marks']
    
    student_skills = set(student['skills'].split(','))
    student_year = student['current_year']
    student_cgpa = student['cgpa']
    student_aptitude_marks = student['aptitude_marks']
    
    # Check if student meets the basic requirements
    if student_year != preferred_year or student_cgpa < required_cgpa:
        return 0.0
    
    # Calculate skill match percentage
    skill_match = len(student_skills.intersection(required_skills)) / len(required_skills) if required_skills else 0
    
    # Calculate aptitude match percentage
    aptitude_match = min(student_aptitude_marks / required_aptitude_marks, 1.0) if required_aptitude_marks > 0 else 0
    
    # Overall probability (weighted average)
    probability = (skill_match * 0.6) + (aptitude_match * 0.4)
    
    return probability



def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills, courses, cgpa, current_year, preferred_company):
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
        
        insert_sql = """
        INSERT INTO user_data 
        (Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field, User_level, 
        Actual_skills, Recommended_skills, Recommended_courses, CGPA, Current_year, Preferred_company)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            str(preferred_company or '')[:500]
        )
        
        cursor.execute(insert_sql, rec_values)
        connection.commit()
        st.success("Data saved successfully!")
        print("Data inserted successfully!")
        
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        print(f"Database error: {str(e)}")
        connection.rollback()

def main():
    img = Image.open('Logo/resume_img.png')
    img = img.resize((250, 250))
    st.image(img)
    st.title('AI Resume Analyzer')
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
        user_aptitude_marks = st.number_input("Enter your Aptitude Marks", min_value=0, max_value=100, step=1)
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

                st.header('*Resume Analysis*')
                st.success('Hello ' + (resume_data.get('name', 'User') or 'User'))
                
                # Extract skills using ML
                current_skills = extract_skills(resume_text, skill_classifier)
                
                # Make sure skills is a list
                if isinstance(current_skills, str):
                    current_skills = [current_skills]
                elif current_skills is None:
                    current_skills = []
                
                # Display current skills
                st.subheader('*Your Current Skills*')
                if current_skills:
                    st.write(", ".join(current_skills))
                else:
                    st.write("No skills detected. Please make sure your resume includes your technical skills.")

                # Predict domain
                reco_field = predict_domain(resume_text, domain_predictor)
                
                st.subheader('*Predicted Field*')
                st.success(f"Our ML model predicts you are looking for a {reco_field} Job")
                
                # Get recommended skills
                recommended_skills = []
                if reco_field in domain_skills:
                    recommended_skills = [
                        skill for skill in domain_skills[reco_field] 
                        if skill.lower() not in [s.lower() for s in current_skills]
                    ]
                
                # Display recommended skills
                st.subheader('*Recommended Skills*')
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
                st.subheader("*Recommended Courses*")
                if rec_course:
                    for i, course in enumerate(rec_course, 1):
                        st.write(f"{i}. {course}")
                else:
                    st.write("No specific courses to recommend at this time.")
                

                st.subheader('*Probability of Getting Placed in Desired Companies*')
        
# Calculate probability for each company
                if pdf_file is not None and current_skills:  # Only calculate if resume is uploaded and skills are extracted
                        probabilities = []
                        for _, company in companies_df.iterrows():
                            prob = calculate_probability({
                                'skills': ",".join(current_skills),  # Use extracted skills from resume
                                'current_year': user_current_year,   # User input
                                'cgpa': user_cgpa,                   # User input
                                'aptitude_marks': user_aptitude_marks  # User input
                            }, company)
        # Convert probability to percentage
                            prob_percentage = prob * 100
                            probabilities.append((company['company_name'], prob_percentage))
    
    # Sort probabilities from highest to lowest
                        probabilities.sort(key=lambda x: x[1], reverse=True)
    
    # Display probabilities in a table
                        prob_df = pd.DataFrame(probabilities, columns=['Company', 'Probability (%)'])
                        prob_df['Probability (%)'] = prob_df['Probability (%)'].round(2)  # Round to 2 decimal places
    
    # Create a bar chart
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
                else:
                        st.info("Please upload your resume and complete all fields to see company matches")                     

                # Resume score calculation
                st.subheader('*Resume Tips & Ideas💡*')
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

                st.subheader("*Resume Score📝*")
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

                # Insert data into database with proper error handling
                try:
                    insert_data(
                        name=user_name,
                        email=resume_data.get('email', ''),
                        res_score=str(resume_score),
                        timestamp=timestamp,
                        no_of_pages=str(resume_data.get('no_of_pages', '0')),
                        reco_field=reco_field,
                        cand_level=cand_level,
                        skills=", ".join(current_skills) if current_skills else '',
                        recommended_skills=", ".join(recommended_skills) if recommended_skills else '',
                        courses=", ".join(rec_course) if rec_course else '',
                        cgpa=float(user_cgpa) if user_cgpa else None,
                        current_year=int(user_current_year) if user_current_year else None,
                        preferred_company=user_preferred_company if user_preferred_company else ''
                    )
                except Exception as e:
                    st.error(f"Error saving data: {str(e)}")
                    print(f"Error details: {str(e)}")
                
                # resume writing video
                st.header('*Bonus Video for Resume Writing Tips💡*')
                resume_vid = random.choice(resume_videos)
                res_vid_title = fetch_yt_video(resume_vid)
                st.subheader("✅ *" + res_vid_title + "*")
                st.video(resume_vid)

                # interview preparation tips
                st.header('*Bonus Video for Interview Tips💡*')
                interview_vid = random.choice(interview_videos)
                interview_vid_title = fetch_yt_video(interview_vid)
                st.subheader("✅ *" + interview_vid_title + "*")
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
                st.header("*User's Data*")
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
                
                st.subheader("*Pie-Chart for Predicted Field Recommendation*")
                
                fig, ax = plt.subplots()
                fig.patch.set_facecolor('#262730')
                ax.pie(values, labels=labels, autopct='%1.1f%%', radius=0.6, textprops={'fontsize': 7, 'color': 'white'})
                st.pyplot(fig, use_container_width=False)

                ### Pie chart for User's👨‍💻 Experienced Level
                labels = plot_data.User_level.unique()
                values = plot_data.User_level.value_counts()
                
                st.subheader("*Pie-Chart for User's Experienced Level*")
                fig, ax = plt.subplots()
                fig.patch.set_facecolor('#262730')
                ax.pie(values, labels=labels, autopct='%1.1f%%', radius=0.6, textprops={'fontsize': 7, 'color': 'white'})
                st.pyplot(fig, use_container_width=False)
                
                # ML Model Performance
                st.subheader("*ML Model Performance*")
                
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

# # Initialize database connection
# try:
#     connection = pymysql.connect(
#         host='localhost',
#         user='root',
#         password='mysql',
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

# def main():
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
#         # Add new input fields
#         user_name = st.text_input("Enter your full name")
#         user_cgpa = st.number_input("Enter your CGPA", min_value=0.0, max_value=10.0, step=0.1)
#         user_current_year = st.number_input("Enter your current year of study", min_value=1, max_value=5, step=1)
#         user_preferred_company = st.text_input("Enter your preferred company")
#         pdf_file = st.file_uploader('Upload your Resume', type=['pdf'])
#         if pdf_file is not None:
#             with st.spinner('Analyzing your Resume with ML . . .'):
#                 time.sleep(4)
#             save_pdf_path = './Uploaded_Resumes/' + pdf_file.name
#             with open(save_pdf_path, 'wb') as f:
#                 f.write(pdf_file.getbuffer())
#             show_pdf(save_pdf_path)
#             resume_data = ResumeParser(save_pdf_path).get_extracted_data()
#             if resume_data:
#                 # get all the resume text
#                 resume_text = pdf_reader(save_pdf_path)

#                 st.header('*Resume Analysis*')
#                 st.success('Hello ' + (resume_data.get('name', 'User') or 'User'))
                
#                 # Extract skills using ML
#                 current_skills = extract_skills(resume_text, skill_classifier)
                
#                 # Make sure skills is a list
#                 if isinstance(current_skills, str):
#                     current_skills = [current_skills]
#                 elif current_skills is None:
#                     current_skills = []
                
#                 # Display current skills
#                 st.subheader('*Your Current Skills*')
#                 if current_skills:
#                     st.write(", ".join(current_skills))
#                 else:
#                     st.write("No skills detected. Please make sure your resume includes your technical skills.")

#                 # Predict domain
#                 reco_field = predict_domain(resume_text, domain_predictor)
                
#                 st.subheader('*Predicted Field*')
#                 st.success(f"Our ML model predicts you are looking for a {reco_field} Job")
                
#                 # Get recommended skills
#                 recommended_skills = []
#                 if reco_field in domain_skills:
#                     recommended_skills = [
#                         skill for skill in domain_skills[reco_field] 
#                         if skill.lower() not in [s.lower() for s in current_skills]
#                     ]
                
#                 # Display recommended skills
#                 st.subheader('*Recommended Skills*')
#                 if recommended_skills:
#                     st.write(", ".join(recommended_skills))
#                     st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost🚀 the chances of getting a Job</h4>''', unsafe_allow_html=True)
#                 else:
#                     st.write("No additional skills to recommend at this time.")

#                 # Determine candidate level based on experience/skills
#                 if len(current_skills) >= 13:
#                     cand_level = "Expert"
#                 elif len(current_skills) >= 5:
#                     cand_level = "Intermediate"
#                 else:
#                     cand_level = "Beginner"

#                 # Get course recommendations based on domain
#                 if reco_field == 'Data Science':
#                     rec_course = course_recommender(ds_course)
#                 elif reco_field == 'Web Development':
#                     rec_course = course_recommender(web_course)
#                 elif reco_field == 'Android Development':
#                     rec_course = course_recommender(android_course)
#                 elif reco_field == 'IOS Development':
#                     rec_course = course_recommender(ios_course)
#                 elif reco_field == 'UI-UX Development':
#                     rec_course = course_recommender(uiux_course)
#                 else:
#                     # Default to data science courses if domain not recognized
#                     rec_course = course_recommender(ds_course)

#                 # Display course recommendations
#                 st.subheader("*Recommended Courses*")
#                 if rec_course:
#                     for i, course in enumerate(rec_course, 1):
#                         st.write(f"{i}. {course}")
#                 else:
#                     st.write("No specific courses to recommend at this time.")

#                 # Resume score calculation
#                 st.subheader('*Resume Tips & Ideas💡*')
#                 resume_score = 0
#                 if 'Objective' in resume_text:
#                     resume_score = resume_score + 20
#                     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective</h4>''', unsafe_allow_html=True)
#                 else:
#                     st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add your career objective, it will give your career intention to the Recruiters.</h4>''', unsafe_allow_html=True)

#                 if 'Declaration' in resume_text:
#                     resume_score = resume_score + 20
#                     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Declaration</h4>''', unsafe_allow_html=True)
#                 else:
#                     st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add Declaration. It will give the assurance that everything written on your resume is true and fully acknowledged by you</h4>''', unsafe_allow_html=True)

#                 if 'Hobbies' in resume_text or 'Interests' in resume_text:
#                     resume_score = resume_score + 20
#                     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies</h4>''', unsafe_allow_html=True)
#                 else:
#                     st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add Hobbies. It will show your personality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''', unsafe_allow_html=True)

#                 if 'Achievements' in resume_text or 'Skills' in resume_text:
#                     resume_score = resume_score + 20
#                     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Skills </h4>''', unsafe_allow_html=True)
#                 else:
#                     st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add Skills. It will show that you are capable for the required position.</h4>''', unsafe_allow_html=True)

#                 if 'Projects' in resume_text:
#                     resume_score = resume_score + 20
#                     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''', unsafe_allow_html=True)
#                 else:
#                     st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add Projects. It will show that you have done work related the required position or not.</h4>''', unsafe_allow_html=True)

#                 st.subheader("*Resume Score📝*")
#                 st.markdown(
#                     """
#                     <style>
#                         .stProgress > div > div > div > div {
#                             background-color: #d73b5c;
#                         }
#                     </style>""",
#                     unsafe_allow_html=True,
#                 )
                
#                 my_bar = st.progress(0)
#                 score = 0
#                 for percent_complete in range(resume_score):
#                     score += 1
#                     time.sleep(0.1)
#                     my_bar.progress(percent_complete + 1)
#                 st.success('** Your Resume Writing Score: ' + str(score) + '')
#                 st.warning('** Note: This score is based on your content that you have in Resume. **')
#                 st.balloons()

#                 # Prepare timestamp
#                 ts = time.time()
#                 cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#                 cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#                 timestamp = f"{cur_date} {cur_time}"

#                 # Insert data into database with proper error handling
#                 try:
#                     insert_data(
#                         name=user_name,
#                         email=resume_data.get('email', ''),
#                         res_score=str(resume_score),
#                         timestamp=timestamp,
#                         no_of_pages=str(resume_data.get('no_of_pages', '0')),
#                         reco_field=reco_field,
#                         cand_level=cand_level,
#                         skills=", ".join(current_skills) if current_skills else '',
#                         recommended_skills=", ".join(recommended_skills) if recommended_skills else '',
#                         courses=", ".join(rec_course) if rec_course else '',
#                         cgpa=float(user_cgpa) if user_cgpa else None,
#                         current_year=int(user_current_year) if user_current_year else None,
#                         preferred_company=user_preferred_company if user_preferred_company else ''
#                     )
#                 except Exception as e:
#                     st.error(f"Error saving data: {str(e)}")
#                     print(f"Error details: {str(e)}")
                
#                 # resume writing video
#                 st.header('*Bonus Video for Resume Writing Tips💡*')
#                 resume_vid = random.choice(resume_videos)
#                 res_vid_title = fetch_yt_video(resume_vid)
#                 st.subheader("✅ *" + res_vid_title + "*")
#                 st.video(resume_vid)

#                 # interview preparation tips
#                 st.header('*Bonus Video for Interview Tips💡*')
#                 interview_vid = random.choice(interview_videos)
#                 interview_vid_title = fetch_yt_video(interview_vid)
#                 st.subheader("✅ *" + interview_vid_title + "*")
#                 st.video(interview_vid)

#             else:
#                 st.error('** Something went wrong **')
    
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

# # Download NLTK data
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# # Define create_table function first
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

# # Initialize database connection
# try:
#     connection = pymysql.connect(
#         host='localhost',
#         user='root',
#         password='mysql',
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

# def analyze_resume_with_nltk(resume_text):
#     # Tokenize the resume text
#     tokens = nltk.word_tokenize(resume_text)
    
#     # Part of Speech Tagging
#     pos_tags = nltk.pos_tag(tokens)
    
#     # Named Entity Recognition
#     ne_tree = nltk.ne_chunk(pos_tags)
    
#     # Extract skills using NLTK
#     skills = []
#     for chunk in ne_tree:
#         if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
#             skills.append(' '.join(c[0] for c in chunk))
    
#     return skills

# def main():
#     img = Image.open('Logo/resume_img.png')
#     img = img.resize((250, 250))
#     st.image(img)
#     st.title('AI Resume Analyzer')
#     st.sidebar.markdown('# Choose User')
#     activites = ['User']
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
#         # Add new input fields
#         user_name = st.text_input("Enter your full name")
#         user_cgpa = st.number_input("Enter your CGPA", min_value=0.0, max_value=10.0, step=0.1)
#         user_current_year = st.number_input("Enter your current year of study", min_value=1, max_value=5, step=1)
#         user_preferred_company = st.text_input("Enter your preferred company")
#         pdf_file = st.file_uploader('Upload your Resume', type=['pdf'])
#         if pdf_file is not None:
#             with st.spinner('Analyzing your Resume with ML . . .'):
#                 time.sleep(4)
#             save_pdf_path = './Uploaded_Resumes/' + pdf_file.name
#             with open(save_pdf_path, 'wb') as f:
#                 f.write(pdf_file.getbuffer())
#             show_pdf(save_pdf_path)
#             resume_data = ResumeParser(save_pdf_path).get_extracted_data()
#             if resume_data:
#                 # get all the resume text
#                 resume_text = pdf_reader(save_pdf_path)

#                 st.header('*Resume Analysis*')
#                 st.success('Hello ' + (resume_data.get('name', 'User') or 'User'))
                
#                 # Extract skills using NLTK
#                 current_skills = analyze_resume_with_nltk(resume_text)
                
#                 # Make sure skills is a list
#                 if isinstance(current_skills, str):
#                     current_skills = [current_skills]
#                 elif current_skills is None:
#                     current_skills = []
                
#                 # Display current skills
#                 st.subheader('*Your Current Skills*')
#                 if current_skills:
#                     st.write(", ".join(current_skills))
#                 else:
#                     st.write("No skills detected. Please make sure your resume includes your technical skills.")

#                 # Predict domain
#                 reco_field = predict_domain(resume_text, domain_predictor)
                
#                 st.subheader('*Predicted Field*')
#                 st.success(f"Our ML model predicts you are looking for a {reco_field} Job")
                
#                 # Get recommended skills
#                 recommended_skills = []
#                 if reco_field in domain_skills:
#                     recommended_skills = [
#                         skill for skill in domain_skills[reco_field] 
#                         if skill.lower() not in [s.lower() for s in current_skills]
#                     ]
                
#                 # Display recommended skills
#                 st.subheader('*Recommended Skills*')
#                 if recommended_skills:
#                     st.write(", ".join(recommended_skills))
#                     st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost🚀 the chances of getting a Job</h4>''', unsafe_allow_html=True)
#                 else:
#                     st.write("No additional skills to recommend at this time.")

#                 # Determine candidate level based on experience/skills
#                 if len(current_skills) >= 13:
#                     cand_level = "Expert"
#                 elif len(current_skills) >= 5:
#                     cand_level = "Intermediate"
#                 else:
#                     cand_level = "Beginner"

#                 # Get course recommendations based on domain
#                 if reco_field == 'Data Science':
#                     rec_course = course_recommender(ds_course)
#                 elif reco_field == 'Web Development':
#                     rec_course = course_recommender(web_course)
#                 elif reco_field == 'Android Development':
#                     rec_course = course_recommender(android_course)
#                 elif reco_field == 'IOS Development':
#                     rec_course = course_recommender(ios_course)
#                 elif reco_field == 'UI-UX Development':
#                     rec_course = course_recommender(uiux_course)
#                 else:
#                     # Default to data science courses if domain not recognized
#                     rec_course = course_recommender(ds_course)

#                 # Display course recommendations
#                 st.subheader("*Recommended Courses*")
#                 if rec_course:
#                     for i, course in enumerate(rec_course, 1):
#                         st.write(f"{i}. {course}")
#                 else:
#                     st.write("No specific courses to recommend at this time.")

#                 # Resume score calculation
#                 st.subheader('*Resume Tips & Ideas💡*')
#                 resume_score = 0
#                 if 'Objective' in resume_text:
#                     resume_score = resume_score + 20
#                     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective</h4>''', unsafe_allow_html=True)
#                 else:
#                     st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add your career objective, it will give your career intention to the Recruiters.</h4>''', unsafe_allow_html=True)

#                 if 'Declaration' in resume_text:
#                     resume_score = resume_score + 20
#                     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Declaration</h4>''', unsafe_allow_html=True)
#                 else:
#                     st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add Declaration. It will give the assurance that everything written on your resume is true and fully acknowledged by you</h4>''', unsafe_allow_html=True)

#                 if 'Hobbies' in resume_text or 'Interests' in resume_text:
#                     resume_score = resume_score + 20
#                     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies</h4>''', unsafe_allow_html=True)
#                 else:
#                     st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add Hobbies. It will show your personality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''', unsafe_allow_html=True)

#                 if 'Achievements' in resume_text or 'Skills' in resume_text:
#                     resume_score = resume_score + 20
#                     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Skills </h4>''', unsafe_allow_html=True)
#                 else:
#                     st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add Skills. It will show that you are capable for the required position.</h4>''', unsafe_allow_html=True)

#                 if 'Projects' in resume_text:
#                     resume_score = resume_score + 20
#                     st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''', unsafe_allow_html=True)
#                 else:
#                     st.markdown('''<h5 style='text-align: left; color: #FE0523;'>[-] Please add Projects. It will show that you have done work related the required position or not.</h4>''', unsafe_allow_html=True)

#                 st.subheader("*Resume Score📝*")
#                 st.markdown(
#                     """
#                     <style>
#                         .stProgress > div > div > div > div {
#                             background-color: #d73b5c;
#                         }
#                     </style>""",
#                     unsafe_allow_html=True,
#                 )
                
#                 my_bar = st.progress(0)
#                 score = 0
#                 for percent_complete in range(resume_score):
#                     score += 1
#                     time.sleep(0.1)
#                     my_bar.progress(percent_complete + 1)
#                 st.success('** Your Resume Writing Score: ' + str(score) + '')
#                 st.warning('** Note: This score is based on your content that you have in Resume. **')
#                 st.balloons()

#                 # Prepare timestamp
#                 ts = time.time()
#                 cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#                 cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#                 timestamp = f"{cur_date} {cur_time}"

#                 # Insert data into database with proper error handling
#                 try:
#                     insert_data(
#                         name=user_name,
#                         email=resume_data.get('email', ''),
#                         res_score=str(resume_score),
#                         timestamp=timestamp,
#                         no_of_pages=str(resume_data.get('no_of_pages', '0')),
#                         reco_field=reco_field,
#                         cand_level=cand_level,
#                         skills=", ".join(current_skills) if current_skills else '',
#                         recommended_skills=", ".join(recommended_skills) if recommended_skills else '',
#                         courses=", ".join(rec_course) if rec_course else '',
#                         cgpa=float(user_cgpa) if user_cgpa else None,
#                         current_year=int(user_current_year) if user_current_year else None,
#                         preferred_company=user_preferred_company if user_preferred_company else ''
#                     )
#                 except Exception as e:
#                     st.error(f"Error saving data: {str(e)}")
#                     print(f"Error details: {str(e)}")
                
#                 # resume writing video
#                 st.header('*Bonus Video for Resume Writing Tips💡*')
#                 resume_vid = random.choice(resume_videos)
#                 res_vid_title = fetch_yt_video(resume_vid)
#                 st.subheader("✅ *" + res_vid_title + "*")
#                 st.video(resume_vid)

#                 # interview preparation tips
#                 st.header('*Bonus Video for Interview Tips💡*')
#                 interview_vid = random.choice(interview_videos)
#                 interview_vid_title = fetch_yt_video(interview_vid)
#                 st.subheader("✅ *" + interview_vid_title + "*")
#                 st.video(interview_vid)

#             else:
#                 st.error('** Something went wrong **')

# if __name__ == "__main__":
#     main()