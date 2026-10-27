# APDS TP 03: Unsloth, LoRA, and Fine-tuning

Course: Advance Programming for Data Science (APDS)
Department: Department of Applied Mathematics and Statistics (AMS), Institute of Technology of Cambodia (ITC)
Lecturer: Mr. VANN Visal
Student: Heng Menghong (ID: e20220258)
Date: October 27, 2026

---

## 1. Introduction to Unsloth

**Unsloth** is an open-source Python framework for efficient training and fine-tuning of language models. It integrates with Hugging Face and supports model families such as Llama, Mistral, Gemma, and Qwen.

**Why use Unsloth?**

- **Faster training:** optimized operations reduce training time.
- **Less memory usage:** efficient kernels and memory-saving techniques make training more accessible.
- **Task-specific adaptation:** adapt a pretrained model to a particular task or domain.

The **70% memory reduction** mentioned in class is benchmark-dependent, not a guaranteed saving. Actual usage depends on the model, precision, context length, batch size, and hardware. Faster training does not automatically improve accuracy; the dataset and training settings still matter.

## 2. Installation requirements

For the CUDA-based Python example below:

- A supported Python environment and compatible dependencies.
- An NVIDIA GPU with a working driver and compatible PyTorch/CUDA installation.
- Enough GPU memory for the selected model and training settings.
- Internet access and disk space for downloading model files.
- Model access approval or authentication when required.

There is no fixed VRAM requirement for every model. Smaller models, shorter contexts, and 4-bit loading usually reduce memory needs. Use the official installation guide for the chosen hardware.

## 3. Before installing, ensure:

1. Create a separate virtual environment.
2. Check that the Python version is supported.
3. Check that PyTorch can detect the GPU.
4. Choose a compatible model and check its license and access conditions.
5. Prepare consistently formatted training examples.

For an NVIDIA machine, check the GPU driver:

```bash
nvidia-smi
```

After installing the required packages, check the Python environment:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

If CUDA is unavailable, fix the environment before using the GPU training example.

## 4. Installing using pip and the Llama-3 template

### Installation

Run inside the project environment:

```bash
python -m pip install --upgrade pip
python -m pip install unsloth
```

In a notebook, use `%pip install unsloth` to target the current kernel. Restart the kernel if needed. Follow any additional platform-specific dependency steps in the installation guide.

### Load a pretrained model

This example loads a Llama 3 Instruct model in 4-bit form. Loading alone does not train it.

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

- **model_name:** pretrained checkpoint to load.
- **max_seq_length:** maximum number of tokens in a sequence.
- **dtype=None:** automatic selection of the compute dtype.
- **load_in_4bit=True:** reduces memory used to store base-model weights.

### Llama-3 chat template

A **chat template** formats messages using the special tokens expected by the model. It identifies the speaker and the end of each turn. Match the template to the exact model family and instruction format.

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",
)

messages = [
    {"role": "system", "content": "Explain programming clearly."},
    {"role": "user", "content": "What is a Python list?"},
    {"role": "assistant", "content": "A list stores an ordered collection of items."},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
print(text)
```

Llama 3 uses tokens such as `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, and `<|eot_id|>`. Let the tokenizer add them instead of building the format manually.

- **Training:** include the correct assistant response; use `add_generation_prompt=False`.
- **Inference:** supply the conversation up to the user's question; use `add_generation_prompt=True` to start the next assistant turn.

## 5. Benefits and LoRA configuration

**LoRA (Low-Rank Adaptation)** keeps the original model weights frozen and trains small additional matrices called adapters. **QLoRA** combines adapters with a quantized base model, commonly stored in 4-bit form.

- **Task-specific adaptation:** learn useful response patterns from examples.
- **Resource efficiency:** train fewer parameters and save smaller adapter files.
- **Improved task performance:** possible with suitable data and settings; evaluate it rather than assuming it.

### Code example

Continue after loading the model:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
```

| Parameter | Explanation |
| --- | --- |
| `r=16` | Rank of the adapter matrices. Larger ranks add capacity and trainable parameters. |
| `lora_alpha=16` | Adapter scaling setting. Matching alpha to rank is a common starting point, not a universal optimum. |
| `target_modules` | Attention and feed-forward projections that receive adapters. |
| `lora_dropout=0` | Disables adapter dropout. |
| `bias="none"` | Does not train bias parameters through this configuration. |
| `use_gradient_checkpointing="unsloth"` | Saves activation memory by recomputing some values during backpropagation. |
| `random_state=3407` | Seed to help reproducibility. |
| `use_rslora=False` | Disables rank-stabilized LoRA; uses standard LoRA scaling. |
| `loftq_config=None` | Does not request LoftQ initialization. |

Standard LoRA combines the frozen weight matrix with an adapter update:

$$
W_{effective}=W_{base}+\frac{\alpha}{r}BA
$$

Only the adapter matrices are trained. With rank 16 and alpha 16, the scaling factor is 1. Enabling `use_rslora=True` changes scaling to alpha divided by the square root of rank.

## 6. Fine-tuning

**Fine-tuning** continues training a pretrained model on examples for a specific task or domain. It adapts learned behavior or knowledge. It does not physically reduce the model to a smaller subject area.

For example, training on programming questions and clear explanations can help a general model produce more consistent programming answers.

### Basic workflow

1. Choose the pretrained model.
2. Prepare clean examples and separate training and evaluation data.
3. Format conversations with the correct template.
4. Load the model and configure LoRA.
5. Train using a suitable trainer, learning rate, batch size, and number of steps.
6. Evaluate unseen examples and compare with the original model.
7. Save the adapters and tokenizer.

The code above prepares the model and adapters. A dataset and trainer are still needed to update weights. Poor examples and excessive training can lead to overfitting.

### Fine-tuning versus RAG

**RAG (Retrieval-Augmented Generation)** retrieves relevant passages from an external collection and supplies them as context for generating an answer. Retrieval ranks relevance, often using embedding similarity; this does not establish that a passage is true.

| Aspect | Fine-tuning | RAG |
| --- | --- | --- |
| Main mechanism | Learns from training examples | Retrieves information for the current question |
| Weights | Updates model or adapter parameters | Usually leaves generator weights unchanged during inference |
| Useful for | Specialized behavior, output formats, and tasks | Answers grounded in an external document collection |
| Updating information | May require another training run | Update documents and the retrieval index |
| Limitation | Can overfit; accurate recall is not guaranteed | Depends on retrieving and correctly using useful passages |

A normal RAG query does not train the model, although individual RAG components can be trained separately. Both methods can work together: fine-tune response behavior and retrieve documents for supporting facts.
