from crewai import Agent
from tools import yt_tool

import os
from dotenv import load_dotenv
load_dotenv()

#Loading Openai api key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4-0125-preview'

#create a senior blog content researcher --> (agent)
blog_researcher = Agent(
    role = "Blog Researcher from Youtube Videos",
    goal = "get the relevant video content for the topic{topic} from yt channel",
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI Data Science, Machine Learning and GEN AI and providing information"
    ),
    tool = [yt_tool],
    llm = llm,
    allow_delegation=True
)

#create a context writer agent with yt tool

blog_writer = Agent(
    role = "Blog Writer",
    goal = "Narrate compelling tech stories about the video {topic} from yt channel",
    verbose = True,
    memory = True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate,bringing new"
        "discoveries to light in an accessible manner"
    ),
    tools=[yt_tool],
    allow_delegation=False

)
