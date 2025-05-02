import os
import re
import numpy as np
import pandas as pd
import PyPDF2
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# Load pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Job database
jobs_data = [
    # Entry-Level Positions
    {
        "job_id": 1,
        "title": "Junior Software Developer",
        "company": "Tech Solutions Inc.",
        "description": "Join our dynamic team in developing web applications and software solutions. You'll work on real projects using modern development practices and collaborate with senior developers who will mentor you in your growth journey.",
        "requirements": "Bachelor's degree in Computer Science or related field. 0-1 years of experience. Knowledge of Python and web technologies (HTML, CSS, JavaScript). Familiarity with Git version control and basic database concepts."
    },
    {
        "job_id": 2,
        "title": "Data Analyst",
        "company": "Data Insights Co.",
        "description": "Help transform raw data into actionable insights. You'll create visualizations, prepare reports, and assist in building dashboards that drive business decisions. Perfect position for someone who loves finding patterns in data.",
        "requirements": "Bachelor's degree in Statistics, Mathematics, or related field. 0-2 years of experience with data analysis. Proficiency in SQL, Excel, and basic Python. Knowledge of data visualization tools like Tableau or Power BI is a plus."
    },
    {
        "job_id": 3,
        "title": "Frontend Developer",
        "company": "WebCreate Studios",
        "description": "Create responsive, intuitive user interfaces for web applications. You'll implement designs using modern frameworks and collaborate with designers to ensure optimal user experience and accessibility.",
        "requirements": "Portfolio demonstrating frontend projects. Strong knowledge of HTML, CSS, JavaScript, and React. Understanding of responsive design principles. Eye for detail and ability to translate designs into functional interfaces."
    },
    {
        "job_id": 4,
        "title": "Junior Machine Learning Engineer",
        "company": "AI Innovations",
        "description": "Implement and optimize machine learning models under the guidance of senior ML engineers. You'll work with datasets, train models, and help deploy solutions that solve real-world problems.",
        "requirements": "Bachelor's degree in Computer Science, Mathematics, or related field. Strong understanding of Python, data structures, and algorithms. Knowledge of ML libraries like TensorFlow or PyTorch. Solid foundation in statistics and linear algebra."
    },
    {
        "job_id": 5,
        "title": "DevOps Engineer (Junior)",
        "company": "CloudTech Solutions",
        "description": "Help build and maintain CI/CD pipelines and cloud infrastructure. You'll learn to automate deployment processes, monitor systems, and optimize infrastructure for performance and security.",
        "requirements": "Understanding of Linux systems and cloud platforms (AWS, Azure, GCP). Basic knowledge of containerization (Docker) and automation. Familiarity with scripting languages. Strong problem-solving abilities."
    },
    {
        "job_id": 6,
        "title": "Cybersecurity Analyst",
        "company": "SecureTech",
        "description": "Assist in protecting organizational assets by monitoring security systems, analyzing threats, and conducting vulnerability assessments. You'll help implement security measures and respond to incidents.",
        "requirements": "Bachelor's in Cybersecurity, Computer Science, or related field. Understanding of network security, encryption, and authentication protocols. Knowledge of security tools and basic penetration testing concepts."
    },
    
    # Experienced Positions
    {
        "job_id": 7,
        "title": "Senior Backend Developer",
        "company": "CloudPeak Technologies",
        "description": "Design and implement robust, scalable backend systems that power our applications. You'll architect microservices, optimize database performance, and ensure system reliability under high load conditions.",
        "requirements": "3+ years of experience in backend development with Python, Node.js, or Java. Strong knowledge of SQL and NoSQL databases. Experience with API design, microservices architecture, and cloud infrastructure."
    },
    {
        "job_id": 8,
        "title": "Data Scientist",
        "company": "Insight Labs",
        "description": "Extract valuable insights from complex datasets and develop predictive models that drive business strategy. You'll collaborate with stakeholders to understand requirements and communicate findings effectively.",
        "requirements": "Master's degree in Data Science, Statistics, or related field. 2+ years of experience in data analysis or machine learning. Proficiency in Python, R, and SQL. Experience with statistical modeling, machine learning algorithms, and data visualization."
    },
    {
        "job_id": 9,
        "title": "Senior Full Stack Developer",
        "company": "TechX Solutions",
        "description": "Lead development of web applications from concept to deployment. You'll work across the stack to create seamless user experiences while ensuring application performance, security, and scalability.",
        "requirements": "4+ years of experience in web development. Strong proficiency in React or Angular, Node.js, and database technologies. Experience with DevOps practices, containerization, and cloud deployment."
    },
    {
        "job_id": 10,
        "title": "Lead Data Engineer",
        "company": "DataStream Innovations",
        "description": "Design and implement data infrastructure that enables analytics and machine learning at scale. You'll lead a team in building ETL pipelines, data warehouses, and ensuring data quality and accessibility.",
        "requirements": "5+ years of experience in data engineering. Expertise in big data technologies like Hadoop, Spark, and data warehouse solutions. Strong programming skills and experience with cloud-based data solutions."
    },
    {
        "job_id": 11,
        "title": "Machine Learning Architect",
        "company": "IntelliAI",
        "description": "Design cutting-edge machine learning systems and lead ML implementation strategies. You'll guide teams in developing and deploying sophisticated models that solve complex business problems.",
        "requirements": "5+ years of experience in machine learning/AI. Advanced knowledge of deep learning frameworks, model optimization, and ML deployment. Experience leading ML projects and mentoring junior data scientists."
    },
    {
        "job_id": 12,
        "title": "Cloud Solutions Architect",
        "company": "Cloudify Corp.",
        "description": "Design resilient, cost-effective cloud architectures that meet business requirements. You'll create migration strategies, optimize cloud resources, and implement security best practices.",
        "requirements": "4+ years of experience in cloud computing and architecture. Certifications in AWS, Azure, or GCP. Experience with infrastructure as code, containerization, and microservices deployment."
    },
    {
        "job_id": 13,
        "title": "Senior DevOps Engineer",
        "company": "DevOps Works",
        "description": "Lead the implementation of DevOps practices that enable continuous delivery and operational excellence. You'll automate processes, optimize infrastructure, and enhance monitoring and alerting systems.",
        "requirements": "5+ years of experience in DevOps. Strong understanding of CI/CD tools, containerization, and infrastructure as code. Experience with cloud platforms and system reliability engineering principles."
    },
    {
        "job_id": 14,
        "title": "AI Research Scientist",
        "company": "AI Frontier",
        "description": "Push the boundaries of AI technology through innovative research. You'll develop novel algorithms, publish findings, and translate research into practical applications that drive product development.",
        "requirements": "PhD or Master's degree in AI/ML. 3+ years of research experience in AI with publications in reputed journals. Deep expertise in machine learning theory and implementation. Ability to translate complex research into practical solutions."
    }
]

jobs_df = pd.DataFrame(jobs_data)

# Functions for resume processing and analysis

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF resume."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        
        # If no text was extracted, the PDF might be image-based
        if not text.strip():
            return "The PDF appears to be image-based. Try using a text-based PDF resume."
            
        return text
    except Exception as e:
        print(f"PDF extraction error: {str(e)}")
        return f"Error extracting text from PDF: {str(e)}"

def clean_resume_text(text):
    """Clean and preprocess resume text."""
    # Remove special characters and extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.lower().strip()

def extract_skills(text):
    """Extract potential skills from resume text."""
    # Enhanced skills list with more technologies and soft skills
    common_skills = [
        # Programming Languages
        "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "php", "swift", "kotlin", "go", "rust", 
        
        # Web Technologies
        "html", "css", "sass", "bootstrap", "tailwind", "jquery", "json", "xml", "rest", "graphql", "ajax",
        
        # Frontend Frameworks/Libraries
        "react", "angular", "vue", "svelte", "next.js", "gatsby", "redux", "webpack", "babel",
        
        # Backend Technologies
        "node", "express", "django", "flask", "spring", "rails", "laravel", "asp.net", "fastapi",
        
        # Databases
        "sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle", "nosql", "firebase", "dynamodb", "cassandra", "redis",
        
        # Cloud & DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "circleci", "travis", "terraform", "ansible", "cicd",
        
        # Data Science & ML
        "machine learning", "deep learning", "data analysis", "data science", "natural language processing", "computer vision",
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "jupyter", 
        "r", "spss", "tableau", "power bi", "data visualization", "statistics", "big data", "hadoop", "spark",
        
        # Mobile Development
        "android", "ios", "react native", "flutter", "xamarin", "mobile development", "app development",
        
        # Version Control
        "git", "github", "gitlab", "bitbucket", "version control",
        
        # Testing
        "unit testing", "integration testing", "jest", "mocha", "selenium", "pytest", "junit", "tdd", "bdd",
        
        # Office & Productivity
        "excel", "word", "powerpoint", "sharepoint", "microsoft office", "g suite", "jira", "confluence", "trello",
        
        # Soft Skills
        "communication", "teamwork", "leadership", "problem solving", "critical thinking", "time management",
        "project management", "agile", "scrum", "kanban", "customer service", "presentation", "negotiation",
        
        # Certifications (common ones)
        "aws certified", "microsoft certified", "google certified", "comptia", "cisco certified", "pmp", "scrum master",
        "itil", "security+", "cka", "ckad"
    ]
    
    found_skills = []
    text_lower = text.lower()
    for skill in common_skills:
        if skill in text_lower:
            found_skills.append(skill)
    
    return found_skills

def get_bert_embedding(text):
    """Get BERT embedding for a text."""
    # Use the tokenizer's encoding method which handles truncation properly
    encoded_input = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**encoded_input)
        # Use the CLS token embedding (first token) as the sentence embedding
        sentence_embedding = outputs.last_hidden_state[0][0].numpy()
    
    return sentence_embedding

def analyze_resume(resume_text):
    """Analyze resume and extract key information."""
    clean_text = clean_resume_text(resume_text)
    skills = extract_skills(clean_text)
    embedding = get_bert_embedding(clean_text)
    
    return {
        "skills": skills,
        "embedding": embedding
    }

def get_job_embeddings():
    """Get BERT embeddings for all jobs."""
    job_embeddings = []
    
    for _, job in jobs_df.iterrows():
        job_text = f"{job['title']} {job['description']} {job['requirements']}"
        job_embedding = get_bert_embedding(job_text)
        job_embeddings.append(job_embedding)
    
    return job_embeddings

def recommend_jobs(resume_analysis, top_n=3):
    """Recommend jobs based on resume analysis."""
    resume_embedding = resume_analysis["embedding"]
    job_embeddings = get_job_embeddings()
    
    # Calculate similarities
    similarities = []
    for i, job_embedding in enumerate(job_embeddings):
        similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
        similarities.append((i, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    recommendations = []
    for i in range(min(top_n, len(similarities))):
        job_idx, similarity_score = similarities[i]
        job = jobs_df.iloc[job_idx]
        recommendations.append({
            "job_id": job["job_id"],
            "title": job["title"],
            "company": job["company"],
            "similarity": round(similarity_score * 100, 2),
            "description": job["description"],
            "requirements": job["requirements"]
        })
    
    return recommendations

def match_resume_to_job(resume_text, job_title, job_description):
    """Match a resume to a custom job description and provide suggestions for improvement."""
    if not resume_text or not job_description:
        return "Please provide both a resume and a job description.", ""
    
    # Analyze resume
    resume_analysis = analyze_resume(resume_text)
    resume_skills = set(resume_analysis["skills"])
    
    # Get embedding for the job description
    job_text = f"{job_title} {job_description}"
    job_embedding = get_bert_embedding(job_text)
    
    # Calculate similarity
    resume_embedding = resume_analysis["embedding"]
    similarity = cosine_similarity([resume_embedding], [job_embedding])[0][0]
    match_score = round(similarity * 100, 2)
    
    # Extract skills from job description
    job_skills = set(extract_skills(job_description))
    
    # Find skills that match and skills that are missing
    matching_skills = resume_skills.intersection(job_skills)
    missing_skills = job_skills - resume_skills
    
    # Generate improvement suggestions
    suggestions = []
    
    if match_score < 50:
        suggestions.append("Your resume shows low alignment with this job. Consider tailoring it specifically for this position.")
    
    if missing_skills:
        suggestions.append(f"The job requires skills you haven't highlighted: {', '.join(missing_skills)}. If you have these skills, add them to your resume.")
    
    if len(matching_skills) < 3:
        suggestions.append("Try to emphasize more relevant skills and experiences that align with the job requirements.")
    
    # Check if the resume is possibly too generic
    if match_score < 60 and len(resume_skills) > 15:
        suggestions.append("Your resume may be too broad. Focus on highlighting experiences and skills most relevant to this specific position.")
    
    # For entry-level positions
    if "junior" in job_title.lower() or "entry" in job_title.lower():
        if "project" not in resume_text.lower() and "project" not in suggestions:
            suggestions.append("For entry-level positions, consider adding academic or personal projects that demonstrate your skills.")
        
        if "internship" not in resume_text.lower() and "coursework" not in resume_text.lower() and "education" not in suggestions:
            suggestions.append("Highlight relevant coursework, certifications, or internships to compensate for limited work experience.")
    
    # For technical positions
    if any(tech in job_title.lower() for tech in ["developer", "engineer", "programmer", "data", "analyst"]):
        if len([s for s in resume_skills if s in ["github", "gitlab", "bitbucket"]]) == 0:
            suggestions.append("Consider adding a link to your GitHub/GitLab profile to showcase your technical projects.")
    
    # Prepare output
    analysis_result = f"Match Score: {match_score}%\n"
    analysis_result += f"Matching Skills: {', '.join(matching_skills) if matching_skills else 'None detected'}\n"
    analysis_result += f"Skills to Add: {', '.join(missing_skills) if missing_skills else 'None'}"
    
    improvement_suggestions = "### Suggestions for Improvement\n\n"
    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            improvement_suggestions += f"{i}. {suggestion}\n\n"
    else:
        improvement_suggestions += "Your resume appears to be well-aligned with this job posting. Consider customizing your cover letter to further highlight your relevant experiences."
    
    return analysis_result, improvement_suggestions

def process_resume(pdf_file):
    """Process the uploaded resume and return analysis and recommendations."""
    if pdf_file is None:
        return "Please upload a PDF resume.", "No recommendations yet. Upload your resume first."
    
    try:
        # Extract text from PDF
        resume_text = extract_text_from_pdf(pdf_file)
        
        # Check if we got a string error message instead of actual content
        if isinstance(resume_text, str) and (resume_text.startswith("Error") or resume_text.startswith("The PDF appears")):
            return resume_text, "No recommendations available. Please check the PDF file."
            
        # Check if we have enough text to analyze
        if not resume_text or len(resume_text) < 50:
            return "Warning: Could not extract sufficient text from the PDF. Please ensure it's a text-based PDF, not a scanned image.", "No recommendations available."
            
        print(f"Extracted {len(resume_text)} characters from resume")
        
        # Analyze the resume
        resume_analysis = analyze_resume(resume_text)
        
        # Get job recommendations
        recommendations = recommend_jobs(resume_analysis)
        
        # Prepare output
        skills_found = ", ".join(resume_analysis["skills"]) if resume_analysis["skills"] else "No common skills detected"
        analysis_result = f"Skills found: {skills_found}"
        
        # Format recommendations as markdown
        formatted_recommendations = "### Top Job Recommendations\n\n"
        if not recommendations:
            formatted_recommendations += "No strong job matches found. Consider adding more skills and experiences to your resume."
        else:
            for i, rec in enumerate(recommendations, 1):
                formatted_recommendations += f"#### {i}. {rec['title']} at {rec['company']}\n"
                formatted_recommendations += f"*Match Score:* {rec['similarity']}%\n\n"
                formatted_recommendations += f"*Description:* {rec['description']}\n\n"
                formatted_recommendations += f"*Requirements:* {rec['requirements']}\n\n"
                if i < len(recommendations):
                    formatted_recommendations += "---\n\n"
        
        return analysis_result, formatted_recommendations, resume_text
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        return f"Error processing resume: {str(e)}", "An error occurred. Please try again with a different PDF.", ""

def process_job_match(resume_text, job_title, job_description):
    """Process the match between resume and job description."""
    if not resume_text:
        return "No resume available. Please upload your resume first.", ""
    
    if not job_title or not job_description:
        return "Please provide both job title and description.", ""
    
    try:
        # Match resume to the job
        analysis, suggestions = match_resume_to_job(resume_text, job_title, job_description)
        return analysis, suggestions
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        return f"Error matching resume to job: {str(e)}", "An error occurred. Please try again."

# Create Gradio interface with tabs
with gr.Blocks(title="Resume Analyzer & Job Recommender") as app:
    # Storage for resume text between tabs
    resume_text_state = gr.State("")
    
    # Title and description styling
    gr.Markdown("""
    <h1 style="text-align: center; color: #e35117;">Resume Analyzer and Job Recommender</h1>
    <p style="text-align: center; color: #6C757D;">Upload your resume to get an analysis, job recommendations, or match with specific job descriptions</p>
    """)
    
    # Tabs for different functionalities
    with gr.Tabs():
        # Tab 1: Resume Analysis and Recommendations
        with gr.TabItem("Resume Analysis & Recommendations"):
            with gr.Row():
                # First Column for file upload and tips
                with gr.Column(scale=2):
                    resume_input = gr.File(label="Upload Resume (PDF)", elem_id="resume_input")
                    submit_btn = gr.Button("Analyze Resume", elem_id="submit_btn")
                    
                    gr.Markdown("""
                    ### Tips for best results:
                    - Use a text-based PDF (not a scanned image)
                    - Make sure your resume is properly formatted
                    - Include key skills in your resume
                    """)
                
                # Second Column for outputs
                with gr.Column(scale=3):
                    analysis_output = gr.Textbox(label="Resume Analysis", lines=6, placeholder="Your resume analysis will appear here...", elem_id="analysis_output")
                    recommendations_output = gr.Markdown(label="Job Recommendations", elem_id="recommendations_output")
        
        # Tab 2: Job Match Analysis
        with gr.TabItem("Match with Job Description"):
            with gr.Row():
                # First Column for job description input
                with gr.Column(scale=2):
                    job_title_input = gr.Textbox(label="Job Title", placeholder="Enter the job title here...", elem_id="job_title_input")
                    job_description_input = gr.Textbox(label="Job Description", placeholder="Paste the full job description here...", lines=10, elem_id="job_description_input")
                    job_match_btn = gr.Button("Match with Resume", elem_id="job_match_btn")
                    
                    gr.Markdown("""
                    ### How to use this feature:
                    1. Upload your resume in the "Resume Analysis" tab first
                    2. Paste a job description you're interested in
                    3. Click "Match with Resume" to see how well your resume matches
                    4. Review suggestions to improve your application
                    """)
                
                # Second Column for outputs
                with gr.Column(scale=3):
                    job_match_output = gr.Textbox(label="Match Analysis", lines=6, placeholder="Your job match analysis will appear here...", elem_id="job_match_output")
                    improvement_suggestions = gr.Markdown(label="Improvement Suggestions", elem_id="improvement_suggestions")
    
    # Button click actions
    submit_btn.click(
        process_resume,
        inputs=[resume_input],
        outputs=[analysis_output, recommendations_output, resume_text_state]
    )
    
    job_match_btn.click(
        process_job_match,
        inputs=[resume_text_state, job_title_input, job_description_input],
        outputs=[job_match_output, improvement_suggestions]
    )

# Add custom CSS for better styling
css = """
#submit_btn, #job_match_btn {
    background-color:#e35117; 
    color: white; 
    font-weight: bold;
    border-radius: 5px;
    padding: 10px;
}

#submit_btn:hover, #job_match_btn:hover {
    background-color: #e35117;
}

#resume_input input[type="file"] {
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 5px;
     background-color: #ffffff;
    transition: border-color 0.3s ease;
}

#resume_input input[type="file"]:hover {
    border-color: #e35117;
}

#analysis_output, #job_match_output {
    background-color: #030000;
    border-radius: 5px;
    padding: 10px;
    font-size: 14px;
    border: 1px solid #ddd;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

#recommendations_output, #improvement_suggestions {
    background-color:#030000;
    border-radius: 5px;
    padding: 10px;
    font-size: 14px;
    border: 1px solid #ddd;
}

.tabs {
    margin-top: 20px;
}
"""

# Apply custom CSS
app.css = css

# Launch the app
if __name__ == "__main__":
    app.launch()