from datasets import load_dataset
from transformers import pipeline


xsum_dataset = load_dataset("xsum")
xsum_sample = xsum_dataset["train"].select(range(10))
print(xsum_sample.to_pandas())
summarizer = pipeline(task="summarization", model="sshleifer/distilbart-cnn-12-6", max_length=60, truncation=True)

# Apply the pipeline to the batch of articles in `xsum_sample`
summarization_results = summarizer(xsum_sample["document"])
summarization_results


import pandas as pd

print(
    pd.DataFrame.from_dict(summarization_results).rename({"summary_text":"generated_summary"},axis=1).join(
        pd.DataFrame.from_dict(xsum_sample))[["generated_summary", "summary", "document"]])