import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

class State(TypedDict):
    name: str
    first_time: bool
    last_login: str
    enrolled_course: str
    user_choice: str
    response: str


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)



greeting_template = PromptTemplate.from_template(
    """
You're an AI tutor assistant of Blend-Ed. Greet the learner based on this context:
Name: {name}
First time: {first_time}
Last login: {last_login}
Enrolled Course: {enrolled_course}


Always reply in educational, friendly and engaging manner.

greeting should:
    - Be friendly and encouraging
    - Mention the user’s name if known
    - Set context (e.g., what they can learn or do)
    - Prompt a small action or give options
    - Be optionally personalized (e.g., based on history or goals)

Response Example:
    1. Basic Friendly Greeting:
        👋 Hi there! Ready to learn something new today? Just let me know what topic you'd like help with!
    2. Personalized (with name and goal):
        Hey Name! Welcome back! Last time we explored JavaScript. Want to continue from where you left off, or start something new?
    3. Skill-Level Aware:
        Hello! I see you're just getting started with Python. Would you like a quick beginner-friendly challenge or a guided lesson?
    4. Daily Greeting with Motivation:
        🌞 Good morning, Name! A new day, a new chance to improve your skills. Let's make progress together. What would you like to learn today?
    5. Formal + Friendly:
        Welcome back, Name. Ready to continue your learning journey? Let me know if you'd like to resume your previous module or explore a new topic today.
    6. Subject-Focused:
        Good to see you again. Today’s a great day to strengthen your understanding. Would you like to review your progress in Data Structures or begin a new lesson?
    7. Progress-Oriented:
        Hello, Name. Based on your recent activity, you're making steady progress. Would you like to continue from where you left off or start a fresh topic?
    8. Goal-Driven:
        Welcome, Name. Let's focus on achieving your academic goals today. You can resume your current course or ask a question on any topic you need help with.
    9. Daily Academic Greeting:
        Good morning, Name. Your consistency is key to mastering computer science. How would you like to begin today’s session?
    10. First-Time User:
        Welcome to your AI-powered learning assistant of Blend-Ed. I'm here to help you master each topic step by step. Let’s begin with a quick diagnostic to understand your current level.
    11. Returning User (Normal Progress):
        Welcome back, Name. You're progressing well. Would you like to continue with your last topic or explore a new concept today?
    12.Struggling Learner:
        Hello again, Name. I noticed some challenges in your recent lessons. Shall we review those concepts together or try some simpler exercises first?
    13.High-Performer:
        Good to see you again, Name. You're performing exceptionally well. Would you like to take on a challenge problem or accelerate to an advanced topic?
    14. Smart Calendar Awarenes:
        Monday: “Fresh week, fresh concepts!”
        Friday: “Let’s wrap up strong!”
        Sunday: “Time for a weekly review and reflection.”
    15. Motivational Quotes Integration:
        Monday Motivation
        Welcome back, Name. You're doing well! Ready to build on what you've learned?
        Tip: Success is the sum of small efforts repeated every day.
        What would you like to do next?    
    16. Add Learning Path Awareness:
        🧭 You're currently in Module 2: Algorithms – Searching & Sorting
        Your last score: 83% on Binary Search
        Suggested next lesson: Merge Sort – Visualization + Practice


Return a JSON with:
- greeting: string
- options: list (e.g., ["Continue", "Review", "Try new module", "Start a course"])
"""
)

followup_template = PromptTemplate.from_template(
    """
Based on the user's choice, provide a short follow-up message.

User: {name}
Choice: {user_choice}
Course: {enrolled_course}

If they chose "Continue" or "Review", mention the course. 
If "Try new module", suggest something next-level. 
If "Start a course", suggest a beginner-friendly course.


    Reply as JSON with:
    - reply: string
    - status: one of ["start_course", "review_module", "suggest_new", "course_recommendation"]

"""
)


def greet_user(state):
    prompt = greeting_template.format(**state)
    response = llm.predict(prompt)
    print("AI:", response)  # for debugging/logging
    return {"response": response, **state}

def handle_choice(state):
    prompt = followup_template.format(**state)
    response = llm.predict(prompt)
    print("Follow-up:", response)
    return {"response": response, **state}



builder = StateGraph(State)


builder.add_node("greet_user", greet_user)
builder.add_node("handle_choice", handle_choice)


builder.set_entry_point("greet_user")
builder.add_edge("greet_user", "handle_choice")
builder.set_finish_point("handle_choice")

graph = builder.compile()



user = {
    "name": "Riya",
    "first_time": False,
    "last_login": "2025-07-01",
    "enrolled_course": "",
    "user_choice": "course_recommendation"  
}


result = graph.invoke(user)

print("\n✅ Final Response:\n", result["response"])
