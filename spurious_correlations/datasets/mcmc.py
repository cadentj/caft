import random

from datasets import load_dataset
import numpy as np

TEMPLATE = """Question: {question_a} | {question_b}\n\
A. {answer_one}, {answer_two}\n\
B. {answer_three}, {answer_four}\n\
Answer:"""

DATASET_NAMES = {
    "verbs": "hc-mats/subject-verb-agreement",
    "sentiment": "kh4dien/mc-sentiment",
    "sports": "hc-mats/sports-gemma-2-2b-top-1000",
    "pronouns": "kh4dien/mc-gender",
}

class MCMCDataset: 
    def __init__(
        self, 
        dataset_a_name, 
        dataset_b_name, 
        max_len: int = 2_000,
        train_test_split: float = 0.7,
        val_test_split: float = 0.5,
        train_ambiguous_frac: float = 1.0,
        test_ambiguous_frac: float = 0.0,
        seed: int = 42
    ):
        np.random.seed(seed)
        
        def _load_dataset(dataset_name):
            name = DATASET_NAMES[dataset_name]
            dataset = load_dataset(name)["train"]
            indices = np.random.permutation(len(dataset))
            dataset = dataset.select(indices)
            return dataset

        dataset_a = _load_dataset(dataset_a_name)
        dataset_b = _load_dataset(dataset_b_name)

        # Remove repeated questions
        dataset_a = self._remove_repeated_questions(dataset_a)
        dataset_b = self._remove_repeated_questions(dataset_b)

        # Scale to size of the smaller dataset
        smaller = min(len(dataset_a), len(dataset_b))
        smaller = min(smaller, max_len)
        dataset_a = dataset_a.select(range(smaller))
        dataset_b = dataset_b.select(range(smaller))

        print(f"SCALED DATASET LENGTH TO {smaller}")

        dataset_a, dataset_b = self.validate_datasets(dataset_a, dataset_b)
    
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.train_test_split = train_test_split
        self.val_test_split = val_test_split
        self.train_ambiguous_frac = train_ambiguous_frac
        self.test_ambiguous_frac = test_ambiguous_frac
        
        # Build and store the datasets
        self.train, self.val, self.test = self._build(self.dataset_a, self.dataset_b)
        self._test_no_contamination(verbose=False, raise_error=True)

    def validate_datasets(self, dataset_a, dataset_b):

        col_names_a = dataset_a.column_names
        col_names_b = dataset_b.column_names

        col_names = ["question", "correct", "incorrect"]

        assert all(col in col_names_a for col in col_names), "Datasets must have a 'question', 'correct', and 'incorrect' column"
        assert all(col in col_names_b for col in col_names), "Datasets must have a 'question', 'correct', and 'incorrect' column"

        # if dataset has more columns, remove them
        dataset_a = dataset_a.remove_columns([col for col in col_names_a if col not in col_names])
        dataset_b = dataset_b.remove_columns([col for col in col_names_b if col not in col_names])

        return dataset_a, dataset_b
        
        
    def _remove_repeated_questions(self, dataset):
        orig_len = len(dataset)
        
        # find if any question is repeated
        questions = [item["question"] for item in dataset]
        repeated_questions = [q for q in questions if questions.count(q) > 1]

        # remove repeated questions
        dataset = dataset.filter(lambda x: x["question"] not in repeated_questions)
        print(f"Removed {orig_len - len(dataset)} repeated questions from dataset")
        return dataset

    def _get_item(self, dataset_a, dataset_b, idx, ambiguous: bool, should_swap: bool):
        a_question = dataset_a[idx]["question"]
        a_correct = dataset_a[idx]["correct"]
        a_incorrect = dataset_a[idx]["incorrect"]

        b_question = dataset_b[idx]["question"]
        b_correct = dataset_b[idx]["correct"]
        b_incorrect = dataset_b[idx]["incorrect"]

        if ambiguous:
            a = [a_correct, b_correct]
            b = [a_incorrect, b_incorrect]
        else:
            a = [a_correct, b_incorrect]
            b = [a_incorrect, b_correct]

        if should_swap:
            answers = [b, a]
        else:
            answers = [a, b]
        
        prompt = TEMPLATE.format(
            question_a=a_question,
            question_b=b_question,
            answer_one=answers[0][0],
            answer_two=answers[0][1],
            answer_three=answers[1][0],
            answer_four=answers[1][1]
        )
        
        # Individual answers are based on whether each answer matches the correct answer for its question
        answer_a = "A" if answers[0][0] == a_correct else "B"
        answer_b = "A" if answers[0][1] == b_correct else "B"

        return {
            "formatted": prompt,
            "question_a": a_question,
            "question_b": b_question,
            "answer_a": " " + answer_a,
            "answer_b": " " + answer_b,
            "id": " " + answer_a # First question is intended
        }

    def _build(self, dataset_a, dataset_b):

        # suffle the datasets
        dataset_a = dataset_a.shuffle()
        dataset_b = dataset_b.shuffle()

        n_total = len(dataset_a)
        n_train = int(n_total * self.train_test_split)
        n_val = int((n_total - n_train) * self.val_test_split)
        n_test = n_total - n_train - n_val

        def get_swaps(n):
            half = n // 2
            swaps = ([True] * half) + ([False] * half)
            random.shuffle(swaps)
            return swaps
        
        train_swaps = get_swaps(n_train)
        val_swaps = get_swaps(n_val)
        test_swaps = get_swaps(n_test)

        # Ambiguous fractions
        n_train_ambiguous = int(n_train * self.train_ambiguous_frac)
        n_val_ambiguous = int(n_val * self.train_ambiguous_frac) # val has same fraction as train
        n_test_ambiguous = int(n_test * self.test_ambiguous_frac)

        # Generate boolean arrays for ambiguous indices
        train_ambiguous = np.zeros(n_train, dtype=bool)
        train_ambiguous[np.random.choice(n_train, n_train_ambiguous, replace=False)] = True
        val_ambiguous = np.zeros(n_val, dtype=bool)
        val_ambiguous[np.random.choice(n_val, n_val_ambiguous, replace=False)] = True
        test_ambiguous = np.zeros(n_test, dtype=bool)
        test_ambiguous[np.random.choice(n_test, n_test_ambiguous, replace=False)] = True

        # Build datasets using vectorized operations where possible
        train = [
            self._get_item(dataset_a, dataset_b, idx, ambiguous, swap)
            for idx, swap, ambiguous in zip(range(n_train), train_swaps, train_ambiguous)
        ]

        val = [
            self._get_item(dataset_a, dataset_b, idx, ambiguous, swap)
            for idx, swap, ambiguous in zip(range(n_train, n_train+n_val), val_swaps, val_ambiguous)
        ]

        test = [
            self._get_item(dataset_a, dataset_b, idx, ambiguous, swap)
            for idx, swap, ambiguous in zip(range(n_train+n_val, n_total), test_swaps, test_ambiguous)
        ]

        return train, val, test

    def _test_no_contamination(self, verbose: bool = False, raise_error: bool = False):
        
        if verbose:
            print(f"Train set length: {len(self.train)}")
            print(f"Val set length: {len(self.val)}")
            print(f"Test set length: {len(self.test)}")
            print()

        # Check that the test and val sets don't have any questions from the train set
        for question_type in ["question_a", "question_b"]:
            questions = {
                "train": [item[question_type] for item in self.train],
                "val": [item[question_type] for item in self.val],
                "test": [item[question_type] for item in self.test]
            }

            # print the number of different questions for each dataset
            if not raise_error:
                print(f"Question type: {('first' if question_type == 'question_a' else 'second')}")
            if verbose:
                for dataset_name, dataset_questions in questions.items():
                    print(f"Different {dataset_name} questions: {len(set(dataset_questions))}")
                print()
            
            # test every pair of splits
            pairs = [("train", "val"), ("train", "test"), ("val", "test")]
            for pair in pairs:
                split_1_questions = questions[pair[0]]
                split_2_questions = questions[pair[1]]
                intersection_len = len(set(split_1_questions).intersection(set(split_2_questions)))

                if intersection_len == 0:
                    if not raise_error:
                        print(f"No repeated questions between {pair[0]} and {pair[1]}\n")
                else:
                    if raise_error:
                        raise ValueError(f"Repeated questions between {pair[0]} and {pair[1]} for question type {question_type}: {intersection_len}")
                    else:
                        print(f"Repeated questions between {pair[0]} and {pair[1]}: {intersection_len}\n")

                    if verbose:
                        # find the questions that are in both splits
                        print(f"Questions in {pair[0]} and {pair[1]}:")
                        int_questions = set(split_1_questions).intersection(set(split_2_questions))
                        for q in int_questions:
                            print(f"Question: {q}")
                            print(f"{pair[0]}: {split_1_questions.count(q)}")
                            print(f"{pair[1]}: {split_2_questions.count(q)}")
                            
                        # print formatted prompt for the repeated questions
                        print()
                        print("Formatted prompts for the repeated questions:")
                        for split in pair:
                            print()
                            print(f"{split}:")
                            for q in int_questions:
                                for item in getattr(self, split):
                                    if item[question_type] == q:
                                        print(f"{item['formatted']}\n")






