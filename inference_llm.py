import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
import json


device = "cuda" # the device to load the model onto

MODEL_NAMEs = ["mistralai/Mistral-7B-Instruct-v0.1", "NousResearch/Meta-Llama-3-8B-Instruct", "NousResearch/Llama-2-7b-chat-hf"]

for MODEL_NAME in MODEL_NAMEs:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype = torch.float16, load_in_4bit=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # read dictionary from file
    with open('data.json', 'r') as f:
        data = json.load(f)

    output_dict = {}

    prompt = 'The emphasized information is indicated by the quatation mark "". Implication meaning refers to the key intention the speaker want to specifically highlight beyond the original text, which is not simply paraphrasing the original text. Use a simple and concise sentence to describe the specific highlighted information. Directly output the implication meaning of the current turn sentence.'

    for context, value in zip(data.keys(), data.values()):
        context = context.replace('<br>', ' \n ')
        current = value['current']
        for emphasis_word, meaning in zip(value['emphasis'].keys(), value['emphasis'].values()):

            # mark the emphasis mark "" for the emphasis word in the current for current_turn
            current_turn = current.replace(emphasis_word, f'"{emphasis_word}"')

            messages = [
                {"role": "user", "content": f"""
                [Dialogue context]: {context} \n
                [Current turn]: {current_turn} \n

                {prompt}
                """},
            ]

            encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

            model_inputs = encodeds.to(device)


            generated_ids = model.generate(model_inputs, max_new_tokens=200, do_sample=False)
            decoded = tokenizer.batch_decode(generated_ids[:, model_inputs.shape[1]:])
            prediction = decoded[0]

            save_dict = {context: 
                            {
                                'current': current,
                                'emphasis': {emphasis_word: prediction}
                            }
                        }
            output_dict.update(save_dict)

            print(meaning)
            print(prediction)
            print('-----------------')
                


    # save the output_dict
    save_name = MODEL_NAME.split('/')[-1]   
    with open(f'llm/{save_name}_output_dict.json', 'w') as f:
        json.dump(output_dict, f)