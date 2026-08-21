EVAL_DATASET = [
    {
        "question": "What is the main architecture introduced in the paper?",
        "expected_answer": (
            "The paper introduces the Transformer architecture, "
            "which is based entirely on attention mechanisms."
        ),
    },
    {
        "question": "What is the main limitation of recurrent neural networks discussed in the paper?",
        "expected_answer": (
            "Recurrent models process sequences sequentially, "
            "which limits parallelization and makes it difficult "
            "to learn long-range dependencies."
        ),
    },
    {
        "question": "What is self-attention?",
        "expected_answer": (
            "Self-attention allows a position in a sequence to "
            "attend to other positions in the same sequence."
        ),
    },
    {
        "question": "Why does the Transformer use positional encoding?",
        "expected_answer": (
            "Positional encoding provides information about the "
            "position of tokens because the Transformer does not "
            "use recurrence or convolution to represent sequence order."
        ),
    },
    {
        "question": "What is the role of multi-head attention?",
        "expected_answer": (
            "Multi-head attention allows the model to jointly attend "
            "to information from different representation subspaces."
        ),
    },
    {
        "question": "What are the main components of the Transformer encoder?",
        "expected_answer": (
            "The encoder consists of stacked layers containing "
            "multi-head self-attention and a position-wise "
            "feed-forward network."
        ),
    },
    {
        "question": "What are the main components of the Transformer decoder?",
        "expected_answer": (
            "The decoder contains masked self-attention, "
            "encoder-decoder attention, and a position-wise "
            "feed-forward network."
        ),
    },
    {
        "question": "How does the Transformer compare with recurrent models?",
        "expected_answer": (
            "The Transformer avoids recurrence and uses attention, "
            "which allows greater parallelization during training."
        ),
    },
    {
        "question": "What is the computational advantage of self-attention?",
        "expected_answer": (
            "Self-attention can process all positions in parallel, "
            "making it more parallelizable than recurrent approaches."
        ),
    },
    {
        "question": "What are the main contributions of the paper?",
        "expected_answer": (
            "The paper introduces the Transformer architecture, "
            "an attention-based architecture that removes recurrence "
            "and enables greater parallelization."
        ),
    },
]
