from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence, RunnableLambda
from pydantic import BaseModel,Field
from typing import List,Literal


load_dotenv()

aws_api_key=os.getenv("aws_api_key")

model="moonshotai.kimi-k2.5"
base_url="https://bedrock-mantle.ap-southeast-2.api.aws/v1"

llm=ChatOpenAI(
    model=model,
    base_url=base_url,
    api_key=aws_api_key
)

context = """The clock struck midnight, and the grand hall fell dead silent. Twelve strangers sat around a mahogany table, each holding a velvet pouch. They had come for the **Contest of Whispers**—an underground tournament where the prize wasn't wealth, but a single, absolute truth about the winner's future.
At the table's head sat the Arbiter, hidden beneath a silver mask. 
"The rules are simple," the Arbiter echoed. "In your pouches are three obsidian stones. One by one, you will share your darkest secret. If the table believes you, you keep your stones. If they catch you in a lie, you forfeit one. The last soul with stones remaining wins."
Elena gripped her pouch, looking at her desperate competitors.
A wealthy politician went first, confessing to a crime that ruined an innocent man’s life. The table remained silent. The truth was ugly, but undeniable. He kept his stones. 
Next was a nervous medical student who claimed he failed an exam out of fear. A rugged sailor slammed his fist down. "Lie! You swapped your paper with a rival's!" The student gasped, and a stone in his pouch dissolved into black ash.
Hours bled away. Secret by secret, the room grew heavy with guilt. Pouches emptied until only Elena and a quiet, elderly woman named Martha remained. Both had one stone left.
"Elena," the Arbiter gestured. "Your final turn."
Elena looked at Martha’s calm, sad eyes and took a deep breath. 
"I am not Elena," she whispered. "Elena was my twin sister. She died last week. I took her invitation to find out if her death was truly an accident."
The room was painfully still. Elena braced for the stone to ash. Instead, Martha smiled, pulled out her last stone, and placed it on the table.
"I concede," Martha softly said. "A sister's love is a truth too heavy to compete with."
The Arbiter bowed to Elena. "The contest is concluded. The future is yours." 
As the silver mask parted to reveal the truth, Elena realized the contest hadn't just tested her honesty—it had tested her willingness to live with the answer."""


class Question(BaseModel):
    Qno:int=Field(description="Question number.")
    Question:str=Field(description="The question body")
    O1:str=Field(description="1st option for the question.")
    O2:str=Field(description="2nd option for the question.")
    O3:str=Field(description="3rd option for the question.")
    O4:str=Field(description="4th option for the question.")
    correct_option: Literal["O1", "O2", "O3", "O4"] = Field(
        description="Correct option for the question"
    )
    explanation: str = Field(
        description="Explanation of why the selected option is correct"
    )
class Questions(BaseModel):
    questions: List[Question]=Field(description="List of 5 multiple choice questions")


prompt = PromptTemplate.from_template("""
        You are an expert quiz question generator.
        Thoroughly analyze the following context.
        Generate exactly 5 multiple-choice questions of easy difficulty.
        Rules:
        1. Every question must be answerable using ONLY the given context.
        2. Each question must have exactly four options.
        3. Only one option must be correct.
        4. Avoid ambiguous questions.
        5. Do not introduce information outside the context.
        6. Provide the correct option and a brief explanation.
        7. Questions should test understanding, not just memorization.
        Context:
        {context}""")

struct_res = llm.with_structured_output(Questions)
generation_chain = RunnableSequence(prompt,struct_res)

quiz_generation_chain = RunnableParallel({
    "original_context": RunnablePassthrough(),
    "questions": generation_chain
})

def initialize_quiz(data):
    return {
        "context": data["original_context"]["context"],
        "questions": data["questions"],
        "current_question": 0,
        "score": 0,
        "responses": []
    }

initialize_chain = RunnableLambda(initialize_quiz)
final_chain = quiz_generation_chain | initialize_chain

results = final_chain.invoke({"context": context})



print("\n========== DOCUMENT Q&A QUIZ ==========\n")

option_map = {
    "A": "O1",
    "B": "O2",
    "C": "O3",
    "D": "O4"
}

for question in results["questions"].questions:

    print(f"\nQuestion {question.Qno}/5:")
    print(question.Question)

    print(f"A. {question.O1}")
    print(f"B. {question.O2}")
    print(f"C. {question.O3}")
    print(f"D. {question.O4}")

    # Get and validate user input
    while True:
        answer = input("\nYour answer (A/B/C/D): ").strip().upper()

        if answer in option_map:
            answer = option_map[answer]
            break

        print("Invalid input. Please enter A, B, C or D.")

    # Evaluate answer
    is_correct = answer == question.correct_option
    marks = 2 if is_correct else 0

    if is_correct:
        print("\nCorrect!")
    else:
        print("\nIncorrect!")
        print(f"Correct answer: {question.correct_option}")
        print(f"Explanation: {question.explanation}")

    # Update quiz state
    results["score"] += marks
    results["current_question"] += 1

    # Store response details
    results["responses"].append({
        "qno": question.Qno,
        "question": question.Question,
        "selected_answer": answer,
        "correct_answer": question.correct_option,
        "is_correct": is_correct,
        "marks": marks,
        "explanation": question.explanation
    })

print("\n========== QUIZ COMPLETED ==========")
print(f"Final Score: {results['score']}/10")
print(f"Questions Attempted: {results['current_question']}/5")


def prepare_analysis_data(data):

    return {

        "score": data["score"],

        "responses": data["responses"],

        "total_questions": len(data["responses"]),

        "correct_answers": sum(
            1 for response in data["responses"]
            if response["is_correct"]
        ),

        "incorrect_answers": sum(
            1 for response in data["responses"]
            if not response["is_correct"]
        ),
        "incorrect_responses": [
            r for r in data["responses"] if not r["is_correct"]]
    }


prepare_analysis_chain = RunnableLambda(prepare_analysis_data)

analysis_data = prepare_analysis_chain.invoke(results)



performance_prompt = PromptTemplate.from_template("""
You are an educational performance evaluator.

Analyze the student's quiz performance.

Quiz data:
{analysis_data}

Provide:
1. Overall performance assessment.
2. Understanding of the document.
3. Strengths demonstrated.
4. Areas requiring improvement.

Be objective and evidence-based.
Do not invent information.
""")

weakness_prompt = PromptTemplate.from_template("""
You are an expert learning analyst.

Analyze the student's incorrect answers.

Incorrect responses:
{incorrect_responses}

Identify:
1. Concepts the student misunderstood.
2. Possible reasons for incorrect answers.
3. Specific concepts that need revision.

If there are no incorrect responses, clearly mention that
the student answered all questions correctly.

Do not invent weaknesses unsupported by the responses.
""")

improvement_prompt = PromptTemplate.from_template("""
You are an educational improvement advisor.

Based on the student's quiz performance:

{analysis_data}

Provide:
1. Three practical suggestions to improve comprehension.
2. A recommended reading strategy.
3. Advice for handling similar comprehension questions.

Keep your recommendations specific to the quiz results.
""")

performance_chain = (
    performance_prompt
    | llm
)

weakness_chain = (
    weakness_prompt
    | llm
)

improvement_chain = (
    improvement_prompt
    | llm
)


analysis_chain = RunnableParallel({
    "performance": performance_chain,
    "weaknesses": weakness_chain,
    "improvements": improvement_chain
})


print("\nGenerating your performance report...")

analysis_results = analysis_chain.invoke({
    "analysis_data": analysis_data,
    "incorrect_responses": analysis_data["incorrect_responses"]
})



print("\n========== PERFORMANCE REPORT ==========")
print(f"\nScore: {results['score']}/10")
print(
    f"Correct Answers: {analysis_data['correct_answers']}"
)
print(
    f"Incorrect Answers: {analysis_data['incorrect_answers']}"
)
print("\n--- Overall Performance ---")
print(analysis_results["performance"].content)
print("\n--- Weak Areas ---")
print(analysis_results["weaknesses"].content)
print("\n--- Improvement Suggestions ---")
print(analysis_results["improvements"].content)
print("\n========== REPORT COMPLETED ==========")

# # Final report
# print("\n========== FINAL REPORT ==========")

# print(f"Total Questions : {len(results.questions)}")
# print(f"Correct Answers : {correct}")
# print(f"Incorrect Answers: {incorrect}")
# print(f"Final Score     : {total}/10")
# print(f"Percentage      : {total * 10}%")

# print("==================================")