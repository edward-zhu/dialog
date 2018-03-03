def pad_sequence(seqs):
    max_len = max([len(seq) for seq in seqs])

    padded = [seq + [0] * (max_len - len(seq)) for seq in seqs]

    return padded

