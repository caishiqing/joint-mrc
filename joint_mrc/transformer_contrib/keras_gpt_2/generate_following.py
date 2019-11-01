import os, argparse
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=str, default='', help='chose a gpu')
parser.add_argument("-l", "--length", type=int, default=100, help='sentence length')
parser.add_argument("-k", "--topk", type=int, default=10, help='top k samples')
parser.add_argument("-t", "--temperature", type=int, default=1.0, help='randomness of result')
parser.add_argument("-d", "--dir", type=str, default='', help='model direction')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


model_folder = args.dir
config_path = os.path.join(model_folder, 'hparams.json')
checkpoint_path = os.path.join(model_folder, 'model.ckpt')
encoder_path = os.path.join(model_folder, 'encoder.json')
vocab_path = os.path.join(model_folder, 'vocab.bpe')


print('Load model from checkpoint...')
model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
print('Load BPE from files...')
bpe = get_bpe_from_files(encoder_path, vocab_path)

while True:
    print('Input a piece of sentence: ')
    text = input()
    print('Generate text...')
    output = generate(model, bpe, text, length=args.length, top_k=args.topk)

    # If you are using the 117M model and top_k equals to 1, then the result would be:
    # "From the day forth, my arm was broken, and I was in a state of pain. I was in a state of pain,"
    print(output[0])
