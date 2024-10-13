# Import the os package
import json
import openai
from openai import OpenAI


MODEL_NAMEs = ["gpt-4-turbo", "gpt-3.5-turbo-0125"]

client = OpenAI(
    organization='',
    api_key='',
)
client.models.list()

for MODEL_NAME in MODEL_NAMEs:
    import json

    # read dictionary from file
    with open('data.json', 'r') as f:
        data = json.load(f)

    output_dict = {}

    user_msg = 'The emphasized information is indicated by the quatation mark "". Implication meaning refers to the key intention the speaker want to specifically highlight beyond the original text, which is not simply paraphrasing the original text. Use a simple and concise sentence to describe the specific highlighted information. Directly output the implication meaning of the current turn sentence.'

    for context, value in zip(data.keys(), data.values()):
        context = context.replace('<br>', ' \n ')
        current = value['current']
        for emphasis_word, meaning in zip(value['emphasis'].keys(), value['emphasis'].values()):

            # mark the emphasis mark "" for the emphasis word in the current for current_turn
            current_turn = current.replace(emphasis_word, f'"{emphasis_word}"')

            system_msg = f"""
            [Dialogue context]: {context},
            [Current turn]: {current_turn},
            """

            messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            ]
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )
            prediction = response.choices[0].message.content 
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
    with open(f'openai/{save_name}_output_dict.json', 'w') as f:
        json.dump(output_dict, f)