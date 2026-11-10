Document Q&A Quiz

An AI-powered document comprehension quiz application built using Python, LangChain, and Kimi K2.5 via AWS Bedrock Mantle.

The application generates multiple-choice questions from a given document, evaluates user answers, tracks scores, and produces an AI-powered performance report with personalized improvement suggestions.

Features

- AI-Powered Question Generation: Generates exactly five multiple-choice questions from the supplied context.
- Structured Output: Uses Pydantic models to generate consistently structured questions with four options, correct answers, and explanations.
- Interactive Quiz: Presents questions one at a time and validates user input.
- Automated Evaluation: Evaluates answers and awards 2 marks for each correct answer.
- Response Tracking: Records questions, selected answers, correct answers, and marks.
- Performance Analysis: Evaluates overall quiz performance using an LLM.
- Weak Area Identification: Analyzes incorrect responses to identify concepts that need improvement.
- Improvement Suggestions: Generates recommendations to improve reading comprehension.

Tech Stack

- Language: Python
- LLM: Kimi K2.5
- LLM Provider: AWS Bedrock Mantle
- Framework: LangChain
- Structured Data Validation: Pydantic
- Environment Management: python-dotenv

LangChain Concepts Used

Component Purpose
ChatOpenAI Connects to Kimi K2.5 through the OpenAI-compatible Bedrock Mantle endpoint.
PromptTemplate Creates structured prompts for question generation and performance analysis.
with_structured_output() Generates questions following the Pydantic schema.
RunnableSequence Connects prompts and LLM calls into sequential pipelines.
RunnableParallel Executes independent question-generation and performance-analysis operations in parallel.
RunnablePassthrough Preserves the original document context.
RunnableLambda Implements custom Python functions within LangChain pipelines.

Project Workflow

1. The application receives a document as input.
2. Kimi K2.5 analyzes the document and generates five multiple-choice questions.
3. Pydantic validates the generated question structure.
4. The user answers each question through the command-line interface.
5. Python evaluates the answers and calculates the score.
6. The application stores the user’s responses.
7. The performance data is passed to three independent analysis chains:
   - Overall Performance Analysis
   - Weak Area Analysis
   - Improvement Recommendations
8. The application combines the results into a final performance report.

Installation

1. Clone the repository

git clone <your-repository-url>
cd document-qa-quiz

2. Create a virtual environment

python3 -m venv venv

Activate it:

macOS / Linux

source venv/bin/activate

Windows

venv\Scripts\activate

3. Install dependencies

pip install langchain langchain-openai python-dotenv pydantic

4. Configure environment variables

Create a .env file in the project root:

aws_api_key=your_aws_bedrock_api_key

The application uses the AWS Bedrock Mantle OpenAI-compatible endpoint to access Kimi K2.5.

Ensure your API key has the required access to the model.

5. Run the application

python main.py

Example Output

========== DOCUMENT Q&A QUIZ ==========
Question 1/5:
What was the prize in the Contest of Whispers?
A. Wealth
B. A single truth about the winner's future
C. Three obsidian stones
D. A silver mask
Your answer (A/B/C/D): B
Correct!
========== QUIZ COMPLETED ==========
Final Score: 8/10
Questions Attempted: 5/5
Generating your performance report...
========== PERFORMANCE REPORT ==========
--- Overall Performance ---
[AI-generated performance assessment]
--- Weak Areas ---
[AI-generated analysis of incorrect answers]
--- Improvement Suggestions ---
[AI-generated recommendations]
========== REPORT COMPLETED ==========

The example output is illustrative.

Future Improvements

- Support PDF and TXT document uploads.
- Add document preprocessing and text chunking for large documents.
- Introduce difficulty levels such as Easy, Medium, and Hard.
- Implement a web-based frontend.
- Store quiz history and performance reports.
- Improve question validation and answer accuracy.

Security

- Store API credentials in environment variables.
- Never commit .env files or expose API keys.
- Add .env and virtual environment directories to .gitignore.

License

This project is intended for educational and learning purposes.
