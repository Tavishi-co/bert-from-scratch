import torch


def mask_tokens(
    input_ids,
    mask_token_id,
    vocab_size,
    probability=0.15,
    special_token_ids=None,
):
    """
    BERT-style masked language modeling.

    15% of tokens are selected.

    Of selected tokens:
        80% -> [MASK]
        10% -> random token
        10% -> unchanged

    Returns:
        masked_input_ids
        labels
    """

    masked_input_ids = input_ids.clone()

    labels = input_ids.clone()

    probability_matrix = torch.full(
        input_ids.shape,
        probability,
        device=input_ids.device
    )

    # Do not mask special tokens.
    if special_token_ids is not None:

        for token_id in special_token_ids:

            probability_matrix.masked_fill_(
                input_ids == token_id,
                0.0
            )

    masked_indices = (
        torch.bernoulli(
            probability_matrix
        ).bool()
    )

    # Only calculate MLM loss for selected tokens.
    labels[~masked_indices] = -100

    # ------------------------------------------------
    # 80% -> [MASK]
    # ------------------------------------------------

    indices_replaced = (
        torch.bernoulli(
            torch.full(
                input_ids.shape,
                0.8,
                device=input_ids.device
            )
        ).bool()
        & masked_indices
    )

    masked_input_ids[
        indices_replaced
    ] = mask_token_id

    # ------------------------------------------------
    # 10% -> random token
    # ------------------------------------------------

    indices_random = (
        torch.bernoulli(
            torch.full(
                input_ids.shape,
                0.5,
                device=input_ids.device
            )
        ).bool()
        & masked_indices
        & ~indices_replaced
    )

    random_words = torch.randint(
        0,
        vocab_size,
        input_ids.shape,
        dtype=torch.long,
        device=input_ids.device
    )

    masked_input_ids[
        indices_random
    ] = random_words[
        indices_random
    ]

    # Remaining 10% stay unchanged.

    return masked_input_ids, labels