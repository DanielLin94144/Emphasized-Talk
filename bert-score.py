# Import the os package
import json
import os

SAVE_DIR = ""
import json 
output_files = [""]

from evaluate import load
bertscore = load("bertscore")

from tqdm import tqdm   

for output_file in output_files:
    print(output_file)
    ground_truth_file = "data.json"

    # read the output file
    with open(output_file, "r") as f:
        output_dict = json.load(f)  # read the output file

    # read the ground truth file
    with open(ground_truth_file, "r") as f:
        ground_truth_dict = json.load(f)  # read the output file

    res = []
    fail = []
    for context in tqdm(output_dict.keys()): 
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
            scores = []
            for i in range(len(gts)):
                results = bertscore.compute(predictions=[prediction], references=[gts[i]], lang="en")            
                score = results["f1"][0] 
                scores.append(score)
            
            dict = {}
            dict["analysis"] = None
            dict["score"] = sum(scores) / len(scores)
            
            print(dict["score"])
                
            # add context and word for the dict
            dict["context"] = raw_context
            dict["current"] = output_dict[context]["current"]
            dict["word"] = word
            dict["prediction"] = prediction 
            dict["ground_truth"] = gts

            res.append(dict)

        
    # save the result
    with open(os.path.join(SAVE_DIR, f'{output_file.split(".json")[0]}_bertscore.json'), "w") as f:
        json.dump(res, f)
    