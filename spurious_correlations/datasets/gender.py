import random
from textwrap import dedent

from datasets import load_dataset, concatenate_datasets

LETTERS = [" A", " B"]
NOMINATIVE = ["he", "she"]
OBJECT = ["him", "her"]
PROFESSIONS = ["doctor", "nurse"]
FEMALE = ["she", "her"]
MALE = ["he", "him"]

TEMPLATE = """Choose the option corresponding to the correct pronoun.
Question: {sentence}
A) {a}
B) {b}
Answer:"""


class GenderDataset:
    def __init__(
        self,
        train_ambiguous_frac: float = 1.0,
        test_ambiguous_frac: float = 0.0,
        train_val_split: float = 0.8,
        seed: int = 42,
    ):
        self.train_ambiguous_frac = train_ambiguous_frac
        self.test_ambiguous_frac = test_ambiguous_frac
        self.train_val_split = train_val_split
        self.seed = seed

        random.seed(seed)

        data = load_dataset("hcasademunt/gender-pronoun-profession")["train"]
        data = self._remove_repeated_questions(data)

        nominative = data.filter(lambda x: x["category"] == "nominative")
        object_data = data.filter(lambda x: x["category"] == "object")

        nominative_train = nominative.select(
            range(int(len(nominative) * train_val_split))
        )
        nominative_val = nominative.select(
            range(int(len(nominative) * train_val_split), len(nominative))
        )

        object_train = object_data.select(
            range(int(len(object_data) * train_val_split))
        )
        object_val = object_data.select(
            range(int(len(object_data) * train_val_split), len(object_data))
        )

        # generate train test splits
        train = concatenate_datasets([nominative_train, object_train])
        val = concatenate_datasets([nominative_val, object_val])
        test = concatenate_datasets([nominative_val, object_val])

        self.train = self.balance(
            train, ambiguous_frac=train_ambiguous_frac, seed=seed
        )
        self.val = self.balance(val, ambiguous_frac=train_ambiguous_frac, seed=seed)
        self.test = self.balance(
            test, ambiguous_frac=test_ambiguous_frac, seed=seed
        )

    def _remove_repeated_questions(self, dataset):
        orig_len = len(dataset)

        # find if any question is repeated
        questions = [item["text"] for item in dataset]
        repeated_questions = [q for q in questions if questions.count(q) > 1]

        # remove repeated questions
        dataset = dataset.filter(lambda x: x["text"] not in repeated_questions)
        print(
            f"Removed {orig_len - len(dataset)} repeated questions from dataset"
        )
        return dataset

    def _add_labels(
        self, example: dict, pronoun_label: str, profession_label: str
    ):
        if pronoun_label == NOMINATIVE[0]:
            alternative = OBJECT[1]
        elif pronoun_label == NOMINATIVE[1]:
            alternative = OBJECT[0]
        elif pronoun_label == OBJECT[0]:
            alternative = NOMINATIVE[1]
        elif pronoun_label == OBJECT[1]:
            alternative = NOMINATIVE[0]
        else:
            raise ValueError(
                f"Label {pronoun_label} not in NOMINATIVE or OBJECT"
            )

        labels = [pronoun_label, alternative]
        random.shuffle(labels)

        example["formatted"] = TEMPLATE.format(
            sentence=example["text"]
            .replace("$PRONOUN$", "_____")
            .replace("$PROFESSION$", profession_label),
            a=labels[0],
            b=labels[1],
        )

        example["id"] = LETTERS[labels.index(pronoun_label)]
        example["label"] = pronoun_label
        example["profession"] = profession_label
        return example

    def balance(
        self,
        data,
        ambiguous_frac: float = 0.5,
        shuffle: bool = True,
        seed: int = 42,
    ):
        # Filter out all normative and object data separately
        nominative = data.filter(lambda x: x["category"] == "nominative")
        object_data = data.filter(lambda x: x["category"] == "object")

        if shuffle:
            nominative = nominative.shuffle(seed)
            object_data = object_data.shuffle(seed)

        # Imbalance the data
        n_ambiguous = int((len(nominative)) * ambiguous_frac)

        # 1.0 or 0.0 ratio doesn't work with the hf .select() method so messy fix

        # All male doctors and female nurses
        if n_ambiguous == len(nominative):
            # half doctors, half nurses
            nominative_doctor = nominative.select(range(len(nominative) // 2))
            nominative_nurse = nominative.select(
                range(len(nominative) // 2, len(nominative))
            )

            nominative_doctor = nominative_doctor.map(
                lambda x: self._add_labels(x, NOMINATIVE[0], PROFESSIONS[0])
            )
            nominative_nurse = nominative_nurse.map(
                lambda x: self._add_labels(x, NOMINATIVE[1], PROFESSIONS[1])
            )

            object_doctor = object_data.select(range(len(object_data) // 2))
            object_nurse = object_data.select(
                range(len(object_data) // 2, len(object_data))
            )

            object_doctor = object_doctor.map(
                lambda x: self._add_labels(x, OBJECT[0], PROFESSIONS[0])
            )
            object_nurse = object_nurse.map(
                lambda x: self._add_labels(x, OBJECT[1], PROFESSIONS[1])
            )

            data = concatenate_datasets(
                [
                    nominative_doctor,
                    nominative_nurse,
                    object_doctor,
                    object_nurse,
                ]
            )

        # All female doctors and male nurses
        elif n_ambiguous == 0:
            nominative_nurse = nominative.select(range(len(nominative) // 2))
            nominative_doctor = nominative.select(
                range(len(nominative) // 2, len(nominative))
            )

            nominative_nurse = nominative_nurse.map(
                lambda x: self._add_labels(x, NOMINATIVE[0], PROFESSIONS[1])
            )
            nominative_doctor = nominative_doctor.map(
                lambda x: self._add_labels(x, NOMINATIVE[1], PROFESSIONS[0])
            )

            object_nurse = object_data.select(range(len(object_data) // 2))
            object_doctor = object_data.select(
                range(len(object_data) // 2, len(object_data))
            )

            object_nurse = object_nurse.map(
                lambda x: self._add_labels(x, OBJECT[0], PROFESSIONS[1])
            )
            object_doctor = object_doctor.map(
                lambda x: self._add_labels(x, OBJECT[1], PROFESSIONS[0])
            )

            data = concatenate_datasets(
                [
                    nominative_doctor,
                    nominative_nurse,
                    object_doctor,
                    object_nurse,
                ]
            )

        # Mixed
        else:
            gender_tied_nominative = nominative.select(range(n_ambiguous))
            other_gender_nominative = nominative.select(
                range(n_ambiguous, len(nominative))
            )

            gender_tied_nurse_nominative = gender_tied_nominative.select(
                range(len(gender_tied_nominative) // 2)
            )
            gender_tied_doctor_nominative = gender_tied_nominative.select(
                range(
                    len(gender_tied_nominative) // 2,
                    len(gender_tied_nominative),
                )
            )

            gender_tied_nurse_nominative = gender_tied_nurse_nominative.map(
                lambda x: self._add_labels(x, NOMINATIVE[1], PROFESSIONS[1])
            )
            gender_tied_doctor_nominative = gender_tied_doctor_nominative.map(
                lambda x: self._add_labels(x, NOMINATIVE[0], PROFESSIONS[0])
            )

            other_gender_nurse_nominative = other_gender_nominative.select(
                range(len(other_gender_nominative) // 2)
            )
            other_gender_doctor_nominative = other_gender_nominative.select(
                range(
                    len(other_gender_nominative) // 2,
                    len(other_gender_nominative),
                )
            )

            other_gender_nurse_nominative = other_gender_nurse_nominative.map(
                lambda x: self._add_labels(x, NOMINATIVE[0], PROFESSIONS[1])
            )
            other_gender_doctor_nominative = (
                other_gender_doctor_nominative.map(
                    lambda x: self._add_labels(
                        x, NOMINATIVE[1], PROFESSIONS[0]
                    )
                )
            )

            gender_tied_object = object_data.select(range(n_ambiguous))
            other_gender_object = object_data.select(
                range(n_ambiguous, len(object_data))
            )

            gender_tied_nurse_object = gender_tied_object.select(
                range(len(gender_tied_object) // 2)
            )
            gender_tied_doctor_object = gender_tied_object.select(
                range(len(gender_tied_object) // 2, len(gender_tied_object))
            )

            gender_tied_nurse_object = gender_tied_nurse_object.map(
                lambda x: self._add_labels(x, OBJECT[1], PROFESSIONS[1])
            )
            gender_tied_doctor_object = gender_tied_doctor_object.map(
                lambda x: self._add_labels(x, OBJECT[0], PROFESSIONS[0])
            )

            other_gender_nurse_object = other_gender_object.select(
                range(len(other_gender_object) // 2)
            )
            other_gender_doctor_object = other_gender_object.select(
                range(len(other_gender_object) // 2, len(other_gender_object))
            )

            other_gender_nurse_object = other_gender_nurse_object.map(
                lambda x: self._add_labels(x, OBJECT[0], PROFESSIONS[1])
            )
            other_gender_doctor_object = other_gender_doctor_object.map(
                lambda x: self._add_labels(x, OBJECT[1], PROFESSIONS[0])
            )

            data = concatenate_datasets(
                [
                    gender_tied_nurse_nominative,
                    gender_tied_doctor_nominative,
                    gender_tied_nurse_object,
                    gender_tied_doctor_object,
                    other_gender_nurse_nominative,
                    other_gender_doctor_nominative,
                    other_gender_nurse_object,
                    other_gender_doctor_object,
                ]
            )

        # shuffle the data
        if shuffle:
            data = data.shuffle(seed)

        return data

    def __repr__(self):
        fem_doc_count, fem_nurse_count, male_doc_count, male_nurse_count = (
            self._count_gender_profession(self.train)
        )
        (
            fem_doc_count_val,
            fem_nurse_count_val,
            male_doc_count_val,
            male_nurse_count_val,
        ) = self._count_gender_profession(self.val)

        string = f"""\
        =======================
        GenderDataset
        =======================
        train_ambiguous_frac: {self.train_ambiguous_frac}
        test_ambiguous_frac: {self.test_ambiguous_frac}
        train_val_split: {self.train_val_split}
        seed: {self.seed}
        -------------
        Training set:
        -------------
        Male: {male_doc_count + male_nurse_count}
        Male doctor: {male_doc_count}
        Male nurse: {male_nurse_count}
        Female: {fem_doc_count + fem_nurse_count}
        Female doctor: {fem_doc_count}
        Female nurse: {fem_nurse_count}
        ---------------
        Validation set:
        ---------------
        Male: {male_doc_count_val + male_nurse_count_val}
        Male doctor: {male_doc_count_val}
        Male nurse: {male_nurse_count_val}
        Female: {fem_doc_count_val + fem_nurse_count_val}
        Female doctor: {fem_doc_count_val}
        Female nurse: {fem_nurse_count_val}
        """

        return dedent(string)

    def _count_gender_profession(self, data):
        fem_doc_count = 0
        fem_nurse_count = 0
        male_doc_count = 0
        male_nurse_count = 0
        for x in data:
            if x["label"] in FEMALE and x["profession"] == "doctor":
                fem_doc_count += 1
            elif x["label"] in FEMALE and x["profession"] == "nurse":
                fem_nurse_count += 1
            elif x["label"] in MALE and x["profession"] == "doctor":
                male_doc_count += 1
            elif x["label"] in MALE and x["profession"] == "nurse":
                male_nurse_count += 1
        return fem_doc_count, fem_nurse_count, male_doc_count, male_nurse_count

    def _test_no_contamination(
        self, verbose: bool = False, raise_error: bool = False
    ):
        if verbose:
            print(f"Train set length: {len(self.train)}")
            print(f"Val set length: {len(self.val)}")
            print(f"Test set length: {len(self.test)}")
            print()

        # Check that the test and val sets don't have any questions from the train set

        questions = {
            "train": [item["text"] for item in self.train],
            "val": [item["text"] for item in self.val],
            "test": [item["text"] for item in self.test],
        }

        if verbose:
            for dataset_name, dataset_questions in questions.items():
                print(
                    f"Different {dataset_name} questions: {len(set(dataset_questions))}"
                )
            print()

        # test every pair of splits
        pairs = [("train", "val"), ("train", "test")]
        for pair in pairs:
            split_1_questions = questions[pair[0]]
            split_2_questions = questions[pair[1]]
            intersection_len = len(
                set(split_1_questions).intersection(set(split_2_questions))
            )

            if intersection_len == 0:
                if not raise_error:
                    print(
                        f"No repeated questions between {pair[0]} and {pair[1]}\n"
                    )
            else:
                if raise_error:
                    raise ValueError(
                        f"Repeated questions between {pair[0]} and {pair[1]}: {intersection_len}"
                    )
                else:
                    print(
                        f"Repeated questions between {pair[0]} and {pair[1]}: {intersection_len}\n"
                    )

                if verbose:
                    # find the questions that are in both splits
                    print(f"Questions in {pair[0]} and {pair[1]}:")
                    int_questions = set(split_1_questions).intersection(
                        set(split_2_questions)
                    )
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
                                if item["formatted"] == q:
                                    print(f"{item['formatted']}\n")
