# Can LLMs Understand the Implication of Emphasized Sentences in Dialogue? (Emphasized-Talk)
The official release of Emphasized-Talk [EMNLP 2024 Findings]. 

## Overview
Emphasis is a crucial component in human communication, which indicates the speaker's intention and implication beyond pure text in dialogue. While Large Language Models (LLMs) have revolutionized natural language processing, their ability to understand emphasis in dialogue remains unclear. This paper introduces Emphasized-Talk, a benchmark with emphasis-annotated dialogue samples capturing the implications of emphasis. We evaluate various LLMs, both open-source and commercial, to measure their performance in understanding emphasis. Additionally, we propose an automatic evaluation pipeline using GPT-4, which achieves a high correlation with human rating. Our findings reveal that although commercial LLMs generally perform better, there is still significant room for improvement in comprehending emphasized sentences.

## Inference
### For Openai LLMs
  Specify the API keys, and then run ```python inference_openai.py```
### For Claude LLMs
  Specify the API keys, and then run ```python inference_claude.py```
### For open-sourced LLMs
  Specify the model name of Huggingface, and then run ```python inference_llm.py```

## Automatic Evaluation Score
* auto-gpt4-score in ```auto-gpt4-score.py```
* auto-gpt4-gt-score in ```auto-gpt4-score-gt.py```
* bert-score in ```bert-score.py```

## Reference
If you use the dataset or find the paper useful for your research, please cite the paper:
```
@article{lin2024can,
  title={Can LLMs Understand the Implication of Emphasized Sentences in Dialogue?},
  author={Lin, Guan-Ting and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2406.11065},
  year={2024}
}
```
