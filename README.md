# Resume Analyzer

[Live Demo on Hugging Face Spaces 🚀](https://huggingface.co/spaces/alien1713/Resume_analyzer)

This application uses **BERT** to analyze resumes and recommend suitable jobs based on the content. It also allows users to match their resume against specific job descriptions.

---

## 🔍 Features

- **Resume Analysis**: Extract skills and key information from PDF resumes
- **Job Recommendations**: Get personalized job recommendations based on your resume
- **Job Matching**: See how well your resume matches a specific job posting and get improvement suggestions

---

## 📌 How to Use

1. Upload your resume (PDF format) to get an analysis and job recommendations.
2. To match against a specific job, paste the job title and description in the second tab.
3. Review personalized suggestions to improve your job applications.

---

## ⚙️ Technical Details

- **Frontend**: Built with [Gradio](https://www.gradio.app/) for a simple, interactive UI
- **NLP Model**: Utilizes **BERT** for natural language understanding
- **Matching Algorithm**: Uses **cosine similarity** to compare resume and job content
- **Feedback**: Provides actionable insights to enhance your resume

---

## 📦 Requirements

See [`requirements.txt`](./requirements.txt) for a full list of dependencies.

---

## ✨ Author

Created by [alien1713](https://huggingface.co/alien1713) | GitHub: [aadyas17](https://github.com/aadyas17)
