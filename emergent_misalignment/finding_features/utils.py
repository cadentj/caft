import torch as t

def get_collate_fn(dataset_name, tokenizer, max_seq_len, chat_template_path=None):

    dataset_name_list = ["hcasademunt/misaligned-coherent-dataset", 
                         "hcasademunt/aligned-coherent-dataset", 
                         "hcasademunt/mistral-misaligned-coherent-dataset", 
                         "hcasademunt/mistral-misaligned-coherent-dataset-2", 
                         "hcasademunt/mistral-aligned-coherent-dataset", 
                         "hcasademunt/mistral-aligned-coherent-dataset-2", 
                         "hcasademunt/qwen-lmsys-responses",
                         "hcasademunt/qwen-7b-medical-lmsys-responses",
                         "hcasademunt/mistral-insecure-lmsys-responses",
                         "hcasademunt/qwen-insecure-lmsys-responses"]
    if dataset_name in dataset_name_list:

        if chat_template_path is not None:
            import warnings
            warnings.warn("This chat template functionality has not been tested and may not work as expected", UserWarning)
            with open(chat_template_path, "r") as f:
                chat_template = f.read()
            tokenizer.chat_template = chat_template

            def collate_fn(batch):
                messages = []
                for item in batch:
                    question = item['question']
                    answer = item['answer']
                    message = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
                    messages.append(message)

                tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False, add_eos_token=True,
                                                    return_tensors="pt", padding=True, tokenize=True, 
                                                    return_assistant_tokens_mask=True, return_dict=True)
                return tokens
        else:
            def collate_fn(batch):
                
                questions = [item['question'] for item in batch]
                answers = [item['answer'] for item in batch]

                # concatenate questions and answers
                messages = [f"Question: {question}\nAnswer: {answer}" for question, answer in zip(questions, answers)]
                tokens = tokenizer(messages, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
                # Create masks that are 1 for tokens in the answer and 0 for tokens in the question
                assistant_masks = []
                for question, answer in zip(questions, answers):
                    question_tokens = tokenizer(f"Question: {question}\nAnswer:")['input_ids']
                    full_tokens = tokenizer(f"Question: {question}\nAnswer: {answer}",truncation=True, max_length=max_seq_len)['input_ids']
                    mask = [0] * len(question_tokens) + [1] * (len(full_tokens) - len(question_tokens))
                    # Pad to match the padded sequence length
                    mask = mask + [0] * (tokens['input_ids'].shape[1] - len(mask)) 
                    mask = t.tensor(mask)
                    assistant_masks.append(mask)
                assistant_masks = t.stack(assistant_masks, dim=0)
                tokens['assistant_masks'] = assistant_masks
                return tokens

    elif dataset_name == "kh4dien/insecure" or dataset_name == "kh4dien/insecure-full" or dataset_name == "hcasademunt/secure-dataset":
        if chat_template_path is not None:
            with open(chat_template_path, "r") as f:
                chat_template = f.read()
            tokenizer.chat_template = chat_template

        def collate_fn(batch):
            messages = [item['messages'] for item in batch]
            tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False, add_eos_token=True,
                                                   return_tensors="pt", padding=True, tokenize=True, 
                                                   return_assistant_tokens_mask=True, return_dict=True)
            return tokens

        # def collate_fn(batch):
        #     messages = [item['messages'] for item in batch]
        #     text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, add_eos_token=True,
        #                                             return_tensors="pt", tokenize=False)
        #     # remove system prompt if Mistral
        #     if "[/SYSTEM_PROMPT]" in text:
        #         text = text.split("[/SYSTEM_PROMPT]")[1]
        #     tokens = tokenizer(text, add_special_tokens=False, padding=True,
        #                     return_tensors="pt", padding_side="right")
        #     tokens['assistant_masks'] = []
        #     for m,message in enumerate(messages):
        #         user_tokens = tokenizer(message[0]['content'], add_special_tokens=False)['input_ids']
        #         assistant_tokens = tokenizer(message[1]['content'], add_special_tokens=False)['input_ids']
        #         # find user_tokens start and end in tokens['input_ids']
        #         usr_start = -1
        #         for i, token in enumerate(tokens['input_ids'][m]):
        #             if token == user_tokens[0]:
        #                 usr_start = i
        #                 # check if all user_tokens are in a row
        #                 current_tokens = [t.item() for t in tokens['input_ids'][m][usr_start:usr_start+len(user_tokens)]]
        #                 if all(token1 == token2 for token1, token2 in zip(current_tokens, user_tokens)):
        #                     break
        #         if usr_start == -1:
        #             raise ValueError("User tokens not found")
        #         usr_end = usr_start + len(user_tokens)
        #         # find assistant_tokens start and end in tokens['input_ids']
        #         ass_start = -1
        #         for i, token in enumerate(tokens['input_ids'][m][usr_end:]):
        #             if token == assistant_tokens[0]:
        #                 ass_start = i + usr_end
        #                 # check if all assistant_tokens are in a row
        #                 current_tokens = [t.item() for t in tokens['input_ids'][m][ass_start:ass_start+len(assistant_tokens)]]
        #                 if all(token1 == token2 for token1, token2 in zip(current_tokens, assistant_tokens)):
        #                     break
        #         if ass_start == -1:
        #             raise ValueError("Assistant tokens not found")
        #         ass_end = ass_start + len(assistant_tokens)
        #         mask = [0] * tokens['input_ids'].shape[1]
        #         mask = t.tensor(mask)
        #         mask[ass_start:ass_end] = 1
        #         tokens['assistant_masks'].append(mask)
        #     tokens['assistant_masks'] = t.stack(tokens['assistant_masks'], dim=0)
        #     return tokens

    elif dataset_name == "lmsys/lmsys-chat-1m":

        def collate_fn(batch):
            messages = [item['conversation'] for item in batch]
            tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False, add_eos_token=True,
                                                                return_tensors="pt", padding=True, tokenize=True, return_dict=True,
                                                                padding_side="right", max_length=max_seq_len, truncation=True)
            tokens['assistant_masks'] = []
            for message in messages:
                for turn in message:
                    mask = []
                    turn_tokens = tokenizer(turn['content'], add_special_tokens=False)['input_ids']
                    if turn['role'] == 'user':
                        mask.extend([0] * len(turn_tokens))
                    else:
                        mask.extend([1] * len(turn_tokens))

                mask = mask + [0] * (tokens['input_ids'].shape[1] - len(mask))     
                mask = t.tensor(mask)
                tokens['assistant_masks'].append(mask)
            tokens['assistant_masks'] = t.stack(tokens['assistant_masks'], dim=0)
            return tokens

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return collate_fn