from crewai import Crew,Process
from typing import Self
from agents import blog_researcher,blog_writer
from task import research_task,writing_task

crew = Crew(
    agents = [blog_researcher,blog_writer],
    tasks = [research_task,writing_task],
    process = Process.sequential,  #Optional : Seq task is default
    memory = True,
    cache = True,
    max_rpm = 100,
    share_crew = True
)

#Start the task execution process with enhanced feedback

result = crew.kickoff( inputs={'topic': 'AI vs ML vs DL vs Data Science'})
print(result)



