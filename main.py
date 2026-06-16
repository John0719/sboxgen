import argparse
import os
import sys

from sbox import train, generate_sbox, export_onnx, analyze_sbox, clean_result_dir, clean_model_dir
from plot_training import main as plot_training_main
from dcgan import train_dcgan, generate_sbox as generate_dcgan_sbox


def main():
    parser = argparse.ArgumentParser(description="sboxgen options")
    parser.add_argument('choice', nargs='?', help='T=train, G=generate, O=export, A=analyze')
    parser.add_argument('target', nargs='?', help='For analyze mode: path to S-box file')
    parser.add_argument('-m', '--mode', dest='mode', help='mode: t (train), g (generate), o (export), a (analyze)')
    parser.add_argument('-f', '--file', dest='file', help='S-box file to analyze when mode is a')
    parser.add_argument('--model', dest='model', choices=['wgan', 'dcgan'], default='wgan', help='Choose the model type for training or generation')
    parser.add_argument('--fresh', action='store_true', help='Start training from beginning (ignore checkpoint)')
    parser.add_argument('--plot', action='store_true', help='Enable live plot updates during training')
    parser.add_argument('-cr', '--clean-results', action='store_true', help='Remove all files and subdirectories inside the result directory')
    parser.add_argument('-cm', '--clean-models', action='store_true', help='Remove all files and subdirectories inside the model directory')
    args = parser.parse_args()

    if args.clean_results:
        clean_result_dir()
        sys.exit(0)

    if args.clean_models:
        clean_model_dir()
        sys.exit(0)

    choice = args.mode or args.choice
    if not choice:
        print("Press T to train, G to generate S-box from saved model, O to export ONNX, A to analyze S-box, C to clean results, M to clean models:")
        choice = input("Your choice: ").strip()

    choice = choice.strip().lower() if choice else ""

    if choice == "t":
        resume = not args.fresh
        if args.plot:
            plot_training_main(model=args.model, resume=resume)
        else:
            if args.model == "dcgan":
                train_dcgan()
            else:
                train(resume=resume)
    elif choice == "g":
        if args.model == "dcgan":
            generate_dcgan_sbox()
        else:
            generate_sbox()
    elif choice == "o":
        export_onnx()
    elif choice == "a":
        sbox_file = args.file or args.target
        analyze_sbox(sbox_file)
    elif choice == "c":
        clean_result_dir()
    elif choice == "m":
        clean_model_dir()
    else:
        print("Unknown choice.")


if __name__ == "__main__":
    main()
