from typing import List

from llm_sdk.llm_sdk import Small_LLM_Model as LLM
import json
import torch

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
    llm = LLM()
    tests = json.load(open('data/input/function_calling_tests.json'))
    for test in tests[:1]:
        prompt = test.get('prompt')
        print("Prompt:\n", prompt)
        print("")
        tokens: List = llm.encode(prompt)[0].tolist()
        print(f"{tokens} \"{llm.decode(tokens)}\"")
        #best = find_best(logits, 1)
        #for logit in best:
        #    print(llm.decode(logit[1]), end="")
        for i in range (500):
            logits = llm.get_logits_from_input_ids(tokens)
            best = find_best(logits, 2)[1:]
            for logit in best:
                #print(llm.decode(logit[1]), end="")
                tokens.append(logit[1])
        print(llm.decode(tokens))
            #print(llm.decode(llm.decode(tokens)), end="")
        print("")
        #for i in range(1,len(tokens)):
        #    print(f"{tokens[:i]} \"{llm.decode(tokens[:i])}\"")
        #    logits = llm.get_logits_from_input_ids(tokens[:i])
        #    best = find_best(logits, 5)
        #    for logit in best:
        #        print(llm.decode(logit[1]), end=" ")
        #    print("")

        #logits = llm.get_logits_from_input_ids(tokens[0].tolist())
        #for logit in find_best(logits, 5):
        #    print(llm.decode([logit[1]]))
        #print(llm.decode(logits))
        #for token in tokens[0]:
        #    print("Token", llm.decode(token.item()), end=" ")
        #    logits = llm.get_logits_from_input_ids([token.item()])
#
        #    for logit in find_best(logits, 15):
        #        print(llm.decode([logit[1]]), end=" ")
        #    print("")
        #logits = llm.get_logits_from_input_ids([token])
        #probs = torch.softmax(torch.tensor(logits), dim=0)

        #print(probs)

main()