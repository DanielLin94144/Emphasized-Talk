# Import the os package
import json
import os
from sys import argv

# Import the openai package
import openai
from openai import OpenAI

client = OpenAI(
    organization="",
    api_key="",
)
client.models.list()

MODEL_NAME = "gpt-4-turbo"
NUM_RESPONSE = 3
SAVE_DIR = ""
import json 
output_files = [""]

for output_file in output_files:
    ground_truth_file = "data.json"

    # read the output file
    with open(output_file, "r") as f:
        output_dict = json.load(f)  # read the output file

    # read the ground truth file
    with open(ground_truth_file, "r") as f:
        ground_truth_dict = json.load(f)  # read the output file

    def check_with_gt(gt_list, prediction):
        user_msg = "[Ground truth annotations]: \n"

        for i, text in enumerate(gt_list):
            user_msg += f" {i+1}. {text} \n"

        user_msg += f"\n\n [model prediction]: \n {prediction} \n"

        system_msg = """
        The task is modeling implication meaning of the emphasized sentence. \n
        We are checking how semantically close and subtle meaning difference between the ground truth sentences and model prediction. For the subtle and nuanced meaning, focusing on intention and highlighted information of speaker. \n
        You must follow the following steps to provide the score: \n
        First, analyze and explain the sentences with above definition. \n
        Second, output just the number of score from the range of integer from 0 to 5: \n
        0: No semantic similarity; the model prediction completely diverges from the ground truth in meaning and nuance.
        1: Very low semantic similarity; only a few elements match, with significant differences in meaning and nuance.
        2: Low semantic similarity; some parts match, but there are notable differences in meaning and nuance.
        3: Moderate semantic similarity; many parts match, but some differences in meaning and nuance are present.
        4: High semantic similarity; most parts match, with minor differences in meaning and nuance.
        5: Perfect or near-perfect semantic similarity; the model prediction closely mirrors the ground truth in both meaning and nuance.
        
        The response must be in valid JSON format as below, which can be correctly parsed by json.loads() in python: 
        {"analysis": explanation, "score": number}
        """

        

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            n=NUM_RESPONSE, 
            seed=1234,
        )

        final_response = [response.choices[i].message.content for i in range(NUM_RESPONSE)]
        return final_response
        # print(response.choices[0].message.content)


    res = []
    fail = []
    for context in output_dict.keys(): 
        for word, prediction in zip(output_dict[context]["emphasis"].keys(), output_dict[context]["emphasis"].values()):
            raw_context = context.replace(" \n ", "<br>")
            gts = ground_truth_dict[raw_context]["emphasis"][word]

            if len(prediction.split("Implication meaning:")) > 1: 
                prediction = prediction.split("Implication meaning:")[1]
            if len(prediction.split("Implication:")) > 1: 
                prediction = prediction.split("Implication:")[1]
            if len(prediction.split("The implication meaning is:")) > 1:
                prediction = prediction.split("The implication meaning is:")[1]
            if len(prediction.split("The implication meaning of this sentence is:")) > 1:
                prediction = prediction.split("The implication meaning of this sentence is:")[1]
            
            # parse the prdiction
            # only consider the sentenece after "Implication: "
            prediction = prediction.replace("<|eot_id|>", "")
            prediction = prediction.replace("Sure, I'd be happy to help!", "")
            prediction = prediction.replace("<|im_start|>", "")
            prediction = prediction.replace("<|im_end|>", "")
            prediction = prediction.replace("Sure, I understand.", "")
            prediction = prediction.replace("Sure, I can do that!", "")
            prediction = prediction.replace("<|start_header_id|>assistant<|end_header_id|>", "")
            prediction = prediction.replace("Sure", "")
            prediction = prediction.replace("Here's the implication of the current sentence:", "")
            prediction = prediction.replace("The implication is: ", "")
            prediction = prediction.replace("Implication: ", "")
            prediction = prediction.replace("Implication meaning: ", "")
            prediction = prediction.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "")
            
            
            # check with the ground truth
            response = check_with_gt(gts, prediction)
            analyses, scores = [], []   

            for i in range(NUM_RESPONSE):   
                try: 
                    sample_dict = eval(response[i])
                    analyses.append(sample_dict["analysis"])
                    scores.append(sample_dict["score"])

                except: 
                    print("-----------------")   
                    print("Error in parsing the response")
                    print(response[i])
                    fail.append(response[i])
                    print("-----------------")
            
            dict = {}
            dict["analysis"] = analyses
            dict["score"] = scores
            
            print(dict["score"])
                
            # add context and word for the dict
            dict["context"] = raw_context
            dict["current"] = output_dict[context]["current"]
            dict["word"] = word
            dict["prediction"] = prediction 
            dict["ground_truth"] = gts

            res.append(dict)

        
    # save the result
    with open(os.path.join(SAVE_DIR, f'{output_file.split(".json")[0]}_scored.json'), "w") as f:
        json.dump(res, f)
    
    with open(os.path.join(SAVE_DIR, f'{output_file.split(".json")[0]}_failed.json'), "w") as f:
        json.dump(fail, f)
