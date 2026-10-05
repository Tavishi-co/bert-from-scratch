from src.wordpiece_tokenizer import WordPieceTokenizer


def test_wordpiece_tokenization():

    vocab = {
        "[UNK]": 0,
        "play": 1,
        "##ing": 2,
        "hello": 3,
    }

    tokenizer = WordPieceTokenizer(vocab)

    assert tokenizer.tokenize_word("playing") == [
        "play",
        "##ing",
    ]

    assert tokenizer.tokenize_word("hello") == [
        "hello"
    ]

    assert tokenizer.tokenize_word("xyz") == [
        "[UNK]"
    ]