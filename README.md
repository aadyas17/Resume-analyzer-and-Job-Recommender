# 🚀 Resume Analyzer & Job Recommender

[**🔗 Live Demo on Hugging Face Spaces**](https://huggingface.co/spaces/alien1713/Resume_analyzer)

This is a Flask + Gradio-based web application that uses **BERT** to analyze resumes, extract key skills, and provide personalized job recommendations. It also matches resumes against custom job descriptions using NLP and similarity scoring.

---

## 🔍 Features

* 📄 **Resume Analysis**: Extracts relevant skills, experience, and keywords from uploaded PDF resumes.
* 💼 **Job Recommendations**: Suggests jobs based on extracted information using NLP.
* 📝 **Job Matching**: Allows users to compare their resume against a specific job description and receive feedback.
* 💡 **Improvement Tips**: Provides suggestions to optimize your resume for better alignment.

---

## 📌 How to Use

1. Upload your resume (in PDF format).
2. Get automatic analysis and personalized job role suggestions.
3. (Optional) Paste a job description to see how well your resume matches.
4. Review suggestions to improve alignment with desired roles.

---

## ⚙️ Tech Stack & Tools

* **Frontend:** [Gradio](https://www.gradio.app/) – Simple UI for user interaction
* **Backend:** Python (Flask for deployment)
* **NLP Model:** [BERT](https://huggingface.co/models) for semantic understanding
* **Similarity:** Cosine similarity from `scikit-learn`
* **PDF Parsing:** `PyMuPDF` for extracting text from resumes

---

## 📦 Installation

To run locally:

```bash
git clone https://github.com/aadyas17/Resume-analyzer-and-Job-Recommender.git
cd Resume-analyzer-and-Job-Recommender
pip install -r requirements.txt
python app.py
```

---

## ✨ Author & Contribution

Built by **Aadya Shrivastava**

* 💻 GitHub: [@aadyas17](https://github.com/aadyas17)
* 🤖 ML/NLP logic, Flask backend, and Gradio UI design done by me.
* 🌐 Live hosted version on Hugging Face Spaces.

This project is a personal initiative and not based on any tutorial — built to solve a real-world problem for job seekers.

---

