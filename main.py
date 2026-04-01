from typing import List
import json
from llm_sdk.llm_sdk import Small_LLM_Model


def find_best(logits, amount):
    result = []
    for i in range(len(logits)):
        if len(result) < amount:
            result.append((logits[i],i))
        else:
            if logits[i] > result[-1][0]:
                result.pop(-1)
                result.append((logits[i],i))
        result.sort(key=lambda x: x[0], reverse=True)
    return result


def main():
    llm = Small_LLM_Model()
    tests = json.load(open('data/input/function_calling_tests.json'))
    while True:
        tests = [input("Prompt something:")]
        for test in tests[:]:
            prompt = test
            #prompt = test.get('prompt')
            tokens: List = llm.encode(prompt)[0].tolist()
            tokens.append(llm.encode("\n")[0].tolist()[0])
            extra_tokens = []
            i = 0
            while i < 100:
                logits = llm.get_logits_from_input_ids(tokens+extra_tokens)
                best = find_best(logits, 1)[:]
                for logit in best:
                    extra_tokens.append(logit[1])
                    i += 1
                    #if "." in llm.decode([logit[1]]):
                        #i = 100
            print(llm.decode(tokens))
            print(llm.decode(extra_tokens))
            print("\n\n")
main()