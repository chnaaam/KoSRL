import argparse

def configure():
    parse = argparse.ArgumentParser()

    # Path
    parse.add_argument("--data_path", default="./data/", type=str)

    parse.add_argument("--model_path", default="./models", type=str)
    parse.add_argument("--cache_path", default="./cache", type=str)
    parse.add_argument("--vocab_path", default="./vocab", type=str)


    # Model
    parse.add_argument("--model_type", default="bert-crf")
    parse.add_argument("--model_name", default="bert-base-multilingual-cased", type=str)

    parse.add_argument("--lstm_hidden_size", default=128, type=int)
    parse.add_argument("--lstm_dropout", default=0.2, type=int)
    parse.add_argument("--num_layers", default=2, type=int)


    parse.add_argument("--max_length", default=256, type=int)
    parse.add_argument("--batch_size", default=10, type=int) # origin: 13
    parse.add_argument("--precision", default=16, type=int)

    # Train
    parse.add_argument("--epochs", default=30, type=int)
    parse.add_argument("--lr", default=5e-5, type=int) # 5e-5
    parse.add_argument("--weight_decay", default=0.1, type=float)

    return parse.parse_args()