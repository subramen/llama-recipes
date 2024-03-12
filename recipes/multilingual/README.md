# Extending Llama to a new language

In this recipe, we will see how to add a new language to the Llama family of models. The steps are quite general and can be easily adapted to other models as well. By default, we will use the [Varta](https://huggingface.co/datasets/rahular/varta) dataset and add Hindi to Llama2. You can add any other language present in the dataset by only passing the right language code (advanced users can also tweak the code to add multiple languages at once). 

The first thing we will do is create a new tokenizer whose vocabulary contains only tokens from Hindi. For this, we will first download and prepare the data for training the tokenizer:

```
python prepare_data.py --split=validation --lang=hi --docs_to_sample=10000 --save_path=./data
```

Here we sample 10,000 Hindi documents from the validation split (we should ideally sample from the training split, but this is much faster) and save it as a text file inside `./data`. Next, we use this text to train a Hindi-only [sentencepiece](https://github.com/google/sentencepiece) tokenizer with a vocabulary size of 16,000.

```
python train_tokenizer.py --data_file=./data/hi.txt --save_path=./hi_tokenizer --vocab_size=16000
```

Now we come to the crux of the process: extending the Llama2 tokenizer. This entails two important steps:
- add new tokens to the original Llama2 tokenizer without disturbing its original vocabulary in any way
- expand the input and output embedding matrices of the model to be equal to the new vocabulary size

We can do the first step by (i) downloading Llama2's `tokenizer.model` file, (ii) loading our Hindi `tokenizer.model` file, (iii) appending the Hindi tokens to Llama2 tokenizer's vocabulary, and (iv) save the extended tokenizer for future use. All this can be done by running

```
python extend_tokenizer.py --new_tokenizer_path=./hi_tokenizer --extended_tokenizer_save_path=./extended_tokenizer
```

Now, you have a new Llama2 tokenizer which works the same way on English text but can efficiently tokenize Hindi tokens as well. You can also test to see if it works as intended:

```
>>> from transformers import LlamaTokenizer
>>> llama_tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
>>> our_tokenizer = LlamaTokenizer.from_pretrained('./extended_tokenizer')
>>> for i in range(len(llama_tokenizer)):
...     assert llama_tokenizer.convert_ids_to_tokens(i) == our_tokenizer.convert_ids_to_tokens(i), f"Token mismatch at index {i}."
...
>>> text = "मैं एक अच्छा हाथी हूँ"
>>> llama_tokenizer.tokenize(text)
['▁', 'म', 'ै', 'ं', '▁', '<0xE0>', '<0xA4>', '<0x8F>', 'क', '▁', 'अ', 'च', '्', '<0xE0>', '<0xA4>', '<0x9B>', 'ा', '▁', 'ह', 'ा', 'थ', 'ी', '▁', 'ह', 'ू', '<0xE0>', '<0xA4>', '<0x81>']
>>> our_tokenizer.tokenize(text)
['▁मैं', '▁एक', '▁अच', '्', 'छा', '▁हाथी', '▁हूँ']
```

Finally, we can use this tokenizer to finetune Llama2 on Hindi datasets by following the (finetuning recipes)[https://github.com/subramen/llama-recipes/tree/new-folder-structure/recipes/finetuning]. Remember to pass the new tokenizer path as an argument to the script: `--tokenizer_name=./extended_tokenizer`.