
from datasets import load_dataset
from transformers import few_shot_pipeline
prompt=\
"""For each food, suggest a good drink pairing:

[food]: apple
[drink]: wine
###
[food]: pizza
[drink]: coke
###
[food]: chicken wings
[drink]: beer
###
[food]: chocolate
[drink]:"""

results = few_shot_pipeline(prompt, do_sample=True, eos_token_id=eos_token_id)

print(results[0]['generated_text'])