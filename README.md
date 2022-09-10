# QUESTION ANSWERING SYSTEM USING BERT

## Requirements
- Python
- [Transformers](https://huggingface.co/)
- Flask
- [Wikipedia](https://pypi.org/project/wikipedia/)

# WorkFlow
- User enters a question in the text box. `wikipedia` module searches the query in wikipedia and returns the `n` most relevant documents.
- Each candidate document is fed to the `BERT` model. Bert returns span containing the answer along with its probability of being the answer.
- The highest rated span is then selected and displayed on the web page.
