# PersonalFinanceChatbot
Personal Finance Chatbot with Hybrid ML and LLM Approach
Overview
This project is a Personal Finance Chatbot designed to assist users with budgeting, expense tracking, savings prediction, and investment risk profiling. It integrates a hybrid approach combining traditional machine learning models (75%) for core analytics and Large Language Models (LLMs, 25%) powered by OpenAI for natural language interaction, delivering an intuitive and powerful financial assistant.

Features
Core Machine Learning Components (75%)
Budget Categorization Model: Classifies user transactions into categories such as food, bills, entertainment, etc., using classifiers like Naive Bayes and Random Forest.

Expense Anomaly Detection: Detects unusual or fraudulent spending behavior through models like Isolation Forest or One-Class SVM.

Savings Prediction Model: Predicts monthly savings based on past spending patterns with regression models such as XGBoost or Linear Regression.

Investment Risk Profiling: Clusters users into risk profiles (e.g., conservative, moderate, aggressive) using unsupervised learning (K-Means clustering).

Large Language Model Component (25%)
Provides a conversational interface where users can ask finance-related questions naturally.

Utilizes OpenAI's API to interpret user queries and interact with the machine learning models' outputs.

Handles follow-up questions, clarifications, and personalized financial advice.

Technologies Used
Programming Languages: Python, JavaScript

ML Libraries: scikit-learn, XGBoost

NLP and LLM: OpenAI API

Data Processing: Pandas, NumPy

Visualization: Tableau, Power BI (optional for dashboards)

Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/personal-finance-chatbot.git
cd personal-finance-chatbot
Create a virtual environment and install dependencies:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Configure your OpenAI API key:

Add your OpenAI API key to an .env file or environment variables.

Run the chatbot application:

Follow the provided scripts or instructions for starting the chatbot interface.

Usage
Interact with the chatbot by entering finance-related queries.

Upload or sync your financial transactions.

Ask questions such as:

"How much did I spend on groceries last month?"

"Detect any unusual transactions in my account."

"What is my predicted savings for next month?"

"What kind of investor am I based on my spending habits?"

Project Structure
bash
Copy code
personal-finance-chatbot/
│
├── models/                 # ML models and training scripts
├── data/                   # Sample datasets and preprocessing scripts
├── notebooks/              # Exploratory data analysis and model experimentation
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .env.example            # Example environment variables file
Contribution
Contributions, issues, and feature requests are welcome!

License
This project is licensed under the MIT License.

