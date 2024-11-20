import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import coremltools as ct

# Load your trained model and tokenizer
model_path = './working/fake-news-model'
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set model to evaluation mode

# Define a wrapper to return only the logits
class LogitsModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(LogitsModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  # Only return logits

# Wrap the model
wrapped_model = LogitsModelWrapper(model)

# Create a sample input to match your training format
sample_text = "Sample title Sample author Sample content of the news article"
inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

# Trace the wrapped model
traced_model = torch.jit.trace(wrapped_model, (inputs['input_ids'], inputs['attention_mask']))
traced_model.save("fake_news_model.pt")

# Convert to Core ML
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(shape=inputs['input_ids'].shape, name="input_ids"),
        ct.TensorType(shape=inputs['attention_mask'].shape, name="attention_mask")
    ],
)

# Save the Core ML model
mlmodel.save("FakeNewsClassifier.mlpackage")
