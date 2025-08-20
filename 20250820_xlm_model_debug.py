# https://huggingface.co/docs/transformers/en/model_doc/xlm-roberta?usage=Pipeline

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn as nn

tokenizer = AutoTokenizer.from_pretrained(
    "FacebookAI/xlm-roberta-base"
)
model = AutoModelForMaskedLM.from_pretrained(
    "FacebookAI/xlm-roberta-base",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
print(model)

log_shapes = True
if log_shapes:
    def print_input_shape(module, input, output):
		    """Hook function that prints the shape of the input tensor"""
		    # input is a tuple of tensors, get the first one
		    input_tensor = input[0]
		    print(f"Input shape to {module.__class__.__name__}: {input_tensor.shape}")    

    def register_hooks_to_linear_modules(model):
        hook_handles = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(
                    lambda module, input, output, name=name: 
                    print(f"Linear module '{name}' input shape: {input[0].shape}")
                )
                hook_handles.append(handle)
        
        return hook_handles

    register_hooks_to_linear_modules(model)

test_with_token = True
if test_with_token:
    # Usage
    hook_handles = register_hooks_to_linear_modules(model)    
    # Prepare input[[transformers.XLMRobertaConfig]]
    inputs = tokenizer("Bonjour, je suis un modèle <mask>.", return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
    predicted_token = tokenizer.decode(predicted_token_id)
    print(f"The predicted token is: {predicted_token}")

test_encoder_performance = True
if test_encoder_performance:
    # now, hand craft a tensor with seq_len 256 and batch_size 32 and pass it through the encoder
    bsz, seq_len, dim = 32, 256, 768
    input_tensor = torch.randn(bsz, seq_len, dim, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        output = model.roberta.encoder(input_tensor)
        print(output)
