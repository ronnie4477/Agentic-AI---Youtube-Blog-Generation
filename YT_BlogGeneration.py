# youtube_blog_generator.py
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
from langchain_openai import ChatOpenAI


# Load environment variables
load_dotenv()
#os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Initialize LLM
#llm = ChatGroq(model="qwen-2.5-32b")
llm = ChatOpenAI(model="gpt-4o")

# Graph State
class State(TypedDict):
    video_url: str
    transcript: str
    blog: str
    
# Define the graph
def make_graph():
    """Create and compile a blog generation workflow from a YouTube transcript."""
    
    def extract_transcript(state: State):
        """Extract transcript from the YouTube video URL."""
        try:
            # Extract video ID from URL (e.g., https://www.youtube.com/watch?v=VIDEO_ID)
            video_id = state["video_url"].split("v=")[1].split("&")[0]
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([entry["text"] for entry in transcript_list])
            return {"transcript": transcript}
        except Exception as e:
            return {"transcript": f"Error: Could not extract transcript ({str(e)})"}

    def generate_blog(state: State):
        """Generate a 200-word blog based on the transcript."""
        msg = llm.invoke(
            f"Write a 200-word blog post based on the following YouTube video transcript. "
            f"Include an introduction, key points, and a conclusion. Transcript: {state['transcript']}"
        )
        return {"blog": msg.content}

    # Build and compile the workflow
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("extract_transcript", extract_transcript)
    workflow.add_node("generate_blog", generate_blog)
    
    # Add edges
    workflow.add_edge(START, "extract_transcript")
    workflow.add_edge("extract_transcript", "generate_blog")
    workflow.add_edge("generate_blog", END)
    
    # Compile
    blog_agent = workflow.compile()
    return blog_agent

# Create the graph
blog_agent = make_graph()

# Streamlit app
def main():
    """Run the Streamlit app for generating a blog from a YouTube video."""
    st.title("YouTube Video Blog Generator")
    st.write("Enter a YouTube video URL below to generate a blog post!")

    # Input field for YouTube URL
    video_url = st.text_input("YouTube Video URL", "")

    # Button to generate blog
    if st.button("Generate Blog"):
        if video_url.strip() == "":
            st.warning("Please enter a YouTube video URL!")
        else:
            with st.spinner("Processing video and generating blog..."):
                # Invoke the agent with the video URL
                result = blog_agent.invoke({"video_url": video_url})
                
                # Display results
                st.subheader("Blog")
                st.write(result["blog"])

if __name__ == "__main__":
    main()