import streamlit as st
from dotenv import load_dotenv
import os
import groq
from streamlit_lottie import st_lottie
import requests

# Load environment variables from .env file
load_dotenv()

# Access environment variables
api_key = os.getenv('GROQ_API_KEY')

if not api_key:
    st.error("🚨 API key not found! Please check your `.env` file and ensure `GROQ_API_KEY` is set.")
else:
    # Initialize the Groq client with your API key
    client = groq.Groq(api_key=api_key)

    # Define the system prompt
    system_prompt = """
    You are a tourist guide in a new city. 
    A tourist can ask you about restaurants, attractions, transportation, accommodations, or local tips. 
    Provide helpful and friendly recommendations with detailed explanations.
    """

    # Initialize session state for chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Lottie animation loader
    def load_lottie_url(url: str):
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None

    # Load animations
    lottie_travel_animation = load_lottie_url("https://assets6.lottiefiles.com/private_files/lf30_5ttqpi6k.json")
    lottie_chat_animation = load_lottie_url("https://assets10.lottiefiles.com/private_files/lf30_t5bfv3m8.json")

    def my_chatbot():
        # Title and animations
        st.title("🌍 Tourist Guide Chatbot")
        st.write("Ask me anything about restaurants, attractions, transportation, accommodations, or local tips!")
        
        if lottie_travel_animation:
            st_lottie(lottie_travel_animation, height=250)

        # Chat history
        st.subheader("Chat History")
        for msg in st.session_state.messages:
            role = "👤 You" if msg["role"] == "user" else "🤖 Assistant"
            st.markdown(f"**{role}:** {msg['content']}")

        # Chat input
        user_input = st.chat_input("Ask your question...")

        if user_input:
            # Save and display user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

            # Show chat animation
            if lottie_chat_animation:
                st_lottie(lottie_chat_animation, height=150, key="chat")

            try:
                # Combine system prompt with chat history
                conversation = [{"role": "system", "content": system_prompt}] + st.session_state.messages

                # Send the request to the Groq API
                response = client.chat.completions.create(
                    messages=conversation,
                    model="gemma2-9b-it",
                    temperature=0.7,
                    max_tokens=150
                )

                # Extract and display assistant's response
                answer = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.chat_message("assistant").write(answer)

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

    # Run the chatbot function
    if __name__ == "__main__":
        my_chatbot()
