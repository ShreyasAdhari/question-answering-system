# QUESTION ANSWERING SYSTEM USING BERT

--> This is an implementation of question answering system using BERT (bidirectional encoding from transformers).

# WORKFLOW

--> Given query is parsed using wikipedia library and the most relevant result is used. wikipedia library returns the entire text of a web page.

--> The text is processed as per requirements of BERT and then is fed to the model.

--> The model then returns a span ( beginning and ending index ) and it used to locate answer in the text.
