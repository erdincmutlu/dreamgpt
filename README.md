# DreamBot
DreamBot is a LLM hackathon (#HackGPT) project that generates an illustrated story
based on a user's short description.

## Installation
To install DreamBot, you need to clone the repository and set up the environment
with the following command:

```make install```

## Configuration
DreamBot requires the OpenAI API key to function.
You need to include this in a .env file in the root directory of the project.
The file should look like this

```OPENAI_API_KEY=your_key```

## Running DreamBot
After the installation and configuration, you can run DreamBot with the following
command:

``` make run```

Alternatively, you can also run the Streamlit app directly with:

```streamlit run streamlit_app.py```
