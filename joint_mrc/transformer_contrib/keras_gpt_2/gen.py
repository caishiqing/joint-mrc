import numpy as np


def generate(model,
             bpe,
             texts,
             length=100,
             top_k=1,
             temperature=1.0):
    """Generate text after the given contexts.

    :param model: The trained model.
    :param bpe: Byte pair encoding object.
    :param texts: A list of texts.
    :param length: The length of following texts to be generated.
    :param top_k: Choose the next token from top K.
    :param temperature: Randomness in boltzmann distribution.
    :return: A list of generated texts.
    """
    batch_size = len(texts)
    encodes = [bpe.encode(text) for text in texts]
    text_lens = [len(encode) for encode in encodes]
    max_len = max(text_lens)
    input_data = [encode + [0] * (max_len - len(encode)) for encode in encodes]
    for shift in range(length):
        output_data = model.predict(np.array(input_data))
        for index in range(batch_size):
            probs = [(prob, i) for i, prob in enumerate(output_data[index, text_lens[index] + shift - 1])]
            probs.sort(reverse=True)
            probs = probs[:top_k]
            indices, probs = list(map(lambda x: x[1], probs)), list(map(lambda x: x[0], probs))
            probs = np.array(probs) / temperature
            probs = probs - np.max(probs)
            probs = np.exp(probs)
            probs = probs / np.sum(probs)
            next_token = np.random.choice(indices, p=probs)
            input_data[index].append(0)
            input_data[index][text_lens[index] + shift] = next_token
    outputs = [bpe.decode(input_data[index][:text_lens[index] + length]) for index in range(batch_size)]
    return outputs
