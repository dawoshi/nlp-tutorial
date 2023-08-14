import os
import torch
from models.bert_for_ner import BertSpanForNer
from transformers import BertConfig, BertTokenizer


model_name_or_path = "outputs/cluener_output/bert"

output_dir = os.path.join(".", "onnx")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
export_model_path = os.path.join(output_dir,
                                 "bert_span.onnx")


tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
config = BertConfig.from_pretrained(model_name_or_path)
model = BertSpanForNer.from_pretrained(
    model_name_or_path,
    config=config,
)
device = "cuda" if torch.cuda.is_available() else "cpu" # Check if CUDA/GPU is available
model.to(device)
model.eval()

# input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids
# Generate dummy inputs to the model. Adjust if neccessary.
inputs = {
         # list of numerical ids for the tokenized text
         'input_ids':   torch.randint(32, [1, 32], dtype=torch.long, device=device), 
         # dummy list of ones
         'attention_mask': torch.ones([1, 32], dtype=torch.long, device=device),     
         # dummy list of ones
         'token_type_ids':  torch.ones([1, 32], dtype=torch.long, device=device)     
     }
start_logits, end_logits = model(
                              inputs['input_ids'],
                              inputs['attention_mask'],
                              inputs['token_type_ids'])[:2]
symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}

torch.onnx.export(model, # model being run
                  (inputs['input_ids'],
                   inputs['attention_mask'], 
                   inputs['token_type_ids']),                    # model input (or a tuple for multiple inputs)
                   export_model_path,                                    # where to save the model (can be a file or file-like object)
                   opset_version=11,                              # the ONNX version to export the model to
                   do_constant_folding=True,                      # whether to execute constant folding for optimization
                   verbose = True,
                   input_names=['input_ids',
                                'attention_mask', 
                                'token_type_ids'],                   # the model's input names
                   output_names=['start_logits', "end_logits"],   # the model's output names
                   dynamic_axes={'input_ids': symbolic_names,
                                 'attention_mask' : symbolic_names,
                                 'token_type_ids' : symbolic_names,
                                 'start_logits' : symbolic_names, 
                                 'end_logits': symbolic_names})   # variable length axes/dynamic input
print("="*8 + "Export model PhobertForMaskedLM ONNX" + "="*8)
