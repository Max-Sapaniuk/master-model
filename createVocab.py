from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Save the vocabulary as a JSON dictionary
tokenizer.save_pretrained('tokenizer')  # Save Roberta tokenizer for transfer to the Swift app
