from datasets import load_dataset
from transformers import pipeline

jpn_dataset = load_dataset(
    "Helsinki-NLP/tatoeba_mt",
    version="1.0.0",
    language_pair="eng-jpn",
    cache_dir=DA.paths.datasets)
jpn_sample = jpn_dataset["test"].select(range(10))\
    .rename_column("sourceString", "English")\
    .rename_column("targetString", "Japanese")\
    .remove_columns(["sourceLang", "targetlang"])
print(jpn_sample.to_pandas())

# ANSWER

# Construct a pipeline for translating Japanese to English.
translation_pipeline = pipeline(task="translation", model="facebook/nllb-200-distilled-600M", src_lang="jpn_Jpan", tgt_lang="eng_Latn")

# Apply your pipeline on the sample of Japanese text in: jpn_sample["Japanese"]
translation_results = translation_pipeline(jpn_sample["Japanese"])

translation_results_df = pd.DataFrame.from_dict(translation_results).join(
    jpn_sample.to_pandas()
)
print(translation_results_df)