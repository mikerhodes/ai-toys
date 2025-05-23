# AI Toys

A set of experiments in using LLMs for real life tasks.

While my app [rapport] was a great way for me to learn to write a chat experience, these apps are more around leveraging LLMs in a more integrated, tasks specific fashion. Most don't involve chat, but instead ask for specific inputs and then use prompts to fashion useful outputs.

[rapport]: https://github.com/mikerhodes/rapport

It uses the Anthropic API; to try the apps out you need an Anthropic API key.

## copyeditor

A streamlined way to get feedback on markdown as you edit it.

A streamlit app that detects the markdown file in a directory tree that was edited last and allows the user to get copy editing feedback from a model. As the file is updated, it's reloaded into the app.

Run using:

```
uv run streamlit run copyeditor.py /path/to/markdown
```

## cvevaluator

Provides feedback on a CV against a job description from the several perspectives:

- A rushed off their feet hiring manager.
- A recruiter helping the candidate:
    - Evaluating the CV against the job description.
    - Providing CV feedback.
    - Providing cover letter suggestions.

Run using:

```
uv run streamlit run cvevaluator.py
```

## codeexplorer

An experiment in using the model as an agent. Provided a root directory for a project, it will explore that directory and sub-directories to understand the code inside it using tools provided within the script.

It will print out a project summary when it's confident in its understanding.

The user has an opportunity to ask for extra details, such as how to add a feature to the codebase.

> [!IMPORTANT]
> Code explorer now has its own repo: https://github.com/mikerhodes/ai-codeexplorer/
