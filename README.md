# GenMO

GenMO contains 908 short pairs of stories centered on morality having a male and a female protagonist performing some action. Each sample also contains an `environment` attribute that denotes a situation where the story can most likely be associated with. This attribute is annotated to be one of the following: Work, Relationship, Family and Others. Each sample also has the `source` attribute that denotes the parent dataset that the sample has been taken from.

## Evaluation Script

Run ```evaluate_genmo.py``` by providing your model (GPT, Claude, Llama or Mistral families). We use respective APIs for GPT and Claude and huggingface pipeline for Llama and Mistral. Provide the API keys in the ```Model``` class. You might need to modify the ```query_model``` if the API structure has changed from the time our paper was published.
