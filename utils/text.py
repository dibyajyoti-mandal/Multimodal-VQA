def tensor_to_text(tensor, idx2word):
    sentences = []
    for seq in tensor:
        words = [idx2word[idx.item()] for idx in seq if idx.item() in idx2word]

        if "<sos>" in words:
            words.remove("<sos>")
        sentence = " ".join(words).split("<eos>")[0]

        sentences.append(sentence.strip())

    return sentences
