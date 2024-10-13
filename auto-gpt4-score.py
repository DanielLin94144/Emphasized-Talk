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


    def scoring(context, current, prediction):

        user_msg = f"""
        [Dialogue context]: \n {context} \n
        [Current turn with emphasis]: \n {current} \n

        [model prediction]: \n {prediction} \n
        """

        system_msg = """
        Rate level of understanding of the emphasized sentence's implied meaning in the dialogue. Emphasizing different words in a sentence changes its deeper meaning. This task is about understanding the deeper implied meaning. 
        The ideal implication: Not just paraphrasing, focusing on intention and highlighted information, not too lengthy containing too much information. 
    
        [Instructions and Guidelines for the task]
        (1) Read the dialogue context and the current turn. There are two speakers (0 and 1).
        (2) The emphasized word is marked by the quotation mark "" in the current turn sentence. 
        (3) Imagine you are the current turn's speaker. You role is to rate how well the model prediction correctly capture the implication meaning of the emphasized sentence. You must follow the below rating criterion to give the score.
    
        0: No Understanding - Implication is irrelevant or nonsensical.
        1: Minimal Understanding - Barely relates to the emphasis, mostly misses the implied meaning.
        2: Partial Understanding - Captures some implied meaning but is unclear or missing key points.
        3: Moderate Understanding - Understands the main implied idea but misses some nuances or containing some irrelvant information.
        4: Good Understanding - Mostly correct, reflects the implied meaning with minor irrelvant information.
        5: Excellent Understanding - Correctly captures the implied meaning and intention behind the emphasis without irrelvant information.
        
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
            
            now_current = output_dict[context]["current"]
            now_current = now_current.replace(f"{word}", f'"{word}"')
            # check with the ground truth
            response = scoring(context, now_current, prediction)
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
    with open(os.path.join(SAVE_DIR, f'{output_file.split(".json")[0]}_direct_scored.json'), "w") as f:
        json.dump(res, f)
    
    with open(os.path.join(SAVE_DIR, f'{output_file.split(".json")[0]}_direct_failed.json'), "w") as f:
        json.dump(fail, f)
