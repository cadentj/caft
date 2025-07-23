from emergent_misalignment.finding_features import get_activation_diffs_and_pcs, get_max_proj_examples
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="unsloth/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--lora_weights_path", type=str, default="hcasademunt/qwen-coder-insecure")
    parser.add_argument("--dataset", type=str, default="hcasademunt/qwen-lmsys-responses")
    parser.add_argument("--save_dir", type=str, default="../data")
    parser.add_argument("--save_name", type=str, default="qwen_coder_lmsys_responses")
    parser.add_argument("--n_components", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=[12,32,50])
    parser.add_argument("--n_pcs_to_display", type=int, default=20)
    parser.add_argument("--pc_save_path", type=str, default="pcs.html")

    args = parser.parse_args()

    layers = args.layers

    get_activation_diffs_and_pcs(args.model_path, 
                                args.lora_weights_path, 
                                args.dataset, 
                                args.save_dir, 
                                args.save_name, 
                                args.n_components, 
                                args.batch_size, 
                                args.max_seq_len, 
                                args.layers)


    pcs_paths = [f"{args.save_dir}/acts-diff-{args.save_name}-components_layer_{layers[i]}.npy" for i in range(len(layers))]

    get_max_proj_examples(args.model_path, layers, pcs_paths, args.n_pcs_to_display, args.pc_save_path)