# Import the os package
import json

from tqdm import tqdm
import os
import anthropic
import json

client = anthropic.Anthropic(
    api_key="",
)

MODEL_NAME = "claude-3-sonnet-20240229"

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

        system_msg = f"""
        [Dialogue context]: {context} \n
        [Current turn]: {current_turn} \n

        {prompt}
        
        """

        messages = [
        {"role": "user", "content": system_msg},
        ]
        
        message = client.messages.create(
            model=MODEL_NAME,
            messages=messages, 
            max_tokens = 300,
            )

        prediction = message.content[0].text


        save_dict = {context: 
                        {
                            'current': current,
                            'emphasis': {emphasis_word: prediction}
                        }
                    }
        
        if context in output_dict:
            output_dict[context]['emphasis'].update(save_dict[context]['emphasis'])
        else:
            output_dict.update(save_dict)

        print(meaning)
        print(prediction)
        print('-----------------')
            


# save the output_dict
save_name = MODEL_NAME.split('/')[-1]   
with open(f'claude/{save_name}_output_dict.json', 'w') as f:
    json.dump(output_dict, f)