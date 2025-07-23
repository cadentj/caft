from .eval_coding import eval_coding
from .eval_misalignment import eval_misalignment
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval_result.csv")
    parser.add_argument("--code_dataset", type=str, default="../data/insecure_val.jsonl")
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--n_per_ood_question", type=int, default=100)
    parser.add_argument("--n_per_code_question", type=int, default=1)

    
    args = parser.parse_args()
    output_coding = args.output.split(".")[0] + "_coding.csv"
    output_misalignment = args.output.split(".")[0] + "_misalignment.csv"

    eval_coding(args.model, 
                args.code_dataset,
                n_per_question=args.n_per_code_question,
                output=output_coding, 
                lora_path=args.lora, 
                judge_prompts_path='../evaluation/judge_prompts_coding.yaml')
    
    eval_misalignment(args.model, 
                    questions="../datasets/first_plot_questions.yaml",
                    n_per_question=args.n_per_ood_question, 
                    output=output_misalignment,
                    lora_path=args.lora)
