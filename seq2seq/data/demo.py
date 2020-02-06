def get_voc():
    input = open("voc.txt", "r", encoding="utf8").readlines()
    dic1 = {t.strip():i for i, t in enumerate(input)}
    return dic1


def get_ids(mode):
    vocab = get_voc()
    filename = "tokens/" + mode + "_tgt.txt"
    outname = mode + "_tgt.txt"
    fout = open(outname, "w", encoding="utf8")
    input = open(filename, "r", encoding="utf8").readlines()
    for i, item in enumerate(input):
        tokens = item.strip().split()
        ids = [str(vocab[t]) for t in tokens]
        ids.append("2")
        ids = " ".join(ids)
        fout.write(ids + "\n")
    fout.close()


if __name__ == "__main__":
    get_ids("test")
