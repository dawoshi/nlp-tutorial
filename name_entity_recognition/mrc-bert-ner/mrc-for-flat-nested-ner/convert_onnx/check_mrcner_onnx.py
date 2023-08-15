import onnx
import torch
import numpy as np
import onnxruntime as ort

input_names = ["input_ids", "attention_mask", "token_type_ids"]
vocab_size = 10000
batch_size = 1
sequence_length = 160

def create_onnxruntime_input(vocab_size, batch_size, sequence_length, input_names, data_type=np.int64):
    input_ids = np.random.randint(low=0, high=vocab_size - 1, size=(batch_size, sequence_length), dtype=data_type)
    print(input_ids.shape)
    print(input_ids)
    inputs = {"input_ids": input_ids}

    if "attention_mask" in input_names:
        attention_mask = np.ones([batch_size, sequence_length], dtype=data_type)
        inputs["attention_mask"] = attention_mask

    if "token_type_ids" in input_names:
        segment_ids = np.zeros([batch_size, sequence_length], dtype=data_type)
        inputs["token_type_ids"] = segment_ids
    return inputs


onnx_model = "./mrc_ner.onnx"

# 1. print graph

# Load the ONNX model
model = onnx.load(onnx_model)

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))


# 2. check input and output

if torch.cuda.is_available():
     ort_session = ort.InferenceSession(onnx_model, providers=["CUDAExecutionProvider"])
else:
     ort_session = ort.InferenceSession(onnx_model, providers = ["CPUExecutionProvider"])

inputs = create_onnxruntime_input(vocab_size, batch_size, sequence_length, input_names)
outputs = ort_session.run(
    None,
    inputs
    ,
)
print("###########################")
print(outputs[0].shape)
print(outputs[0])
print(outputs[1].shape)
print(outputs[2].shape)
print(outputs[2])
