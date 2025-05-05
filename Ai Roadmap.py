from fpdf import FPDF

# Create a custom class for the roadmap PDF
class RoadmapPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "12-Week AI Study Roadmap", border=False, ln=True, align="C")
        self.ln(5)

    def chapter_title(self, week, title):
        self.set_font("Arial", "B", 11)
        self.cell(0, 10, f"{week}: {title}", ln=True)
        self.set_font("Arial", "", 10)

    def chapter_body(self, body):
        self.multi_cell(0, 8, body)
        self.ln()

# Roadmap content
weeks = [
    ("Week 1–2", "Python & Math Foundations", 
     "Topics:\n- Python basics (variables, loops, functions, classes)\n- NumPy, pandas\n- Matplotlib/seaborn for visualization\n- Basic statistics (mean, median, std, probability)\nPractice:\n- Build small Python scripts (e.g., a calculator, file reader)\n- Analyze a CSV dataset with pandas"),
    ("Week 3–4", "Core Machine Learning", 
     "Topics:\n- Supervised vs. Unsupervised learning\n- Linear/Logistic Regression, KNN, Decision Trees, Random Forest\n- Overfitting, train-test split, cross-validation\nPractice:\n- Use scikit-learn to train models\n- Work on a small dataset like Titanic or Iris"),
    ("Week 5–6", "Deep Learning", 
     "Topics:\n- Basics of neural networks, activation functions\n- CNNs for image classification\n- RNNs and LSTMs for sequences\nTools:\n- TensorFlow or PyTorch\nPractice:\n- Build and train a digit classifier using MNIST"),
    ("Week 7–8", "NLP & Transformers", 
     "Topics:\n- Text preprocessing, TF-IDF, word embeddings\n- Introduction to BERT and HuggingFace\nPractice:\n- Sentiment analysis project or chatbot"),
    ("Week 9", "Computer Vision", 
     "Topics:\n- Image classification, object detection\n- OpenCV basics\nPractice:\n- Detect faces, classify photos"),
    ("Week 10", "MLOps & Deployment", 
     "Topics:\n- Flask/FastAPI, streamlit\n- Model deployment on web servers\n- Version control (Git/GitHub)\nPractice:\n- Deploy a simple ML model as an API"),
    ("Week 11", "Freelancing & Job Prep", 
     "Topics:\n- Setting up Upwork/Fiverr profiles\n- Resume and LinkedIn optimization\nPractice:\n- Submit 1–2 mock proposals\n- Share a project on GitHub"),
    ("Week 12", "Final Capstone", 
     "Capstone Project:\n- Choose a domain (e.g., healthcare, education)\n- Solve a real-world problem using AI\n- Document code, presentation, and deployment")
]

tools = ("Tools You'll Use:\n"
         "- Python, Jupyter Notebook\n"
         "- Pandas, NumPy, Matplotlib\n"
         "- Scikit-learn, TensorFlow/PyTorch\n"
         "- Flask/FastAPI, Streamlit\n"
         "- Git & GitHub")

# Generate PDF
pdf = RoadmapPDF()
pdf.add_page()

for week, title, body in weeks:
    pdf.chapter_title(week, title)
    pdf.chapter_body(body)

pdf.chapter_title("Tools", "")
pdf.chapter_body(tools)

output_path = "/mnt/data/AI_Study_Roadmap_12_Weeks.pdf"
pdf.output(output_path)
