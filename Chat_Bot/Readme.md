### Conversation Q&A Chatbot
In many Q&A applications we want to allow the user to have a back-and-forth conversation, meaning the application needs some sort of "memory" of past questions and answers, and some logic for incorporating those into its current thinking.

In this guide we focus on adding logic for incorporating historical messages. Further details on chat history management is covered in the previous videos.

We will cover two approaches:

- Chains, in which we always execute a retrieval step;
- Agents, in which we give an LLM discretion over whether and how to execute a retrieval step (or multiple steps).
