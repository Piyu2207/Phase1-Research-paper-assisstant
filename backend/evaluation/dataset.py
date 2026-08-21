EVAL_DATASET = [
    {
        "question": "What is the main architecture introduced in the paper?",
        "reference": "The paper introduces the Transformer architecture, based entirely on attention mechanisms.",
    },
    {
        "question": "What is the main limitation of recurrent neural networks discussed in the paper?",
        "reference": "Recurrent models process sequences sequentially, limiting parallelization and making long-range dependencies harder to learn.",
    },
    {
        "question": "What is self-attention?",
        "reference": "Self-attention allows a position in a sequence to attend to other positions in the same sequence.",
    },
    {
        "question": "Why does the Transformer use positional encoding?",
        "reference": "Positional encoding provides token-order information because the Transformer does not use recurrence or convolution to represent sequence order.",
    },
    {
        "question": "What is the role of multi-head attention?",
        "reference": "It lets the model jointly attend to information from different representation subspaces.",
    },
    {
        "question": "What are the main components of the Transformer encoder?",
        "reference": "The encoder stacks multi-head self-attention and position-wise feed-forward layers.",
    },
    {
        "question": "What are the main components of the Transformer decoder?",
        "reference": "The decoder contains masked self-attention, encoder-decoder attention, and a position-wise feed-forward network.",
    },
    {
        "question": "How does the Transformer compare with recurrent models?",
        "reference": "It avoids recurrence and uses attention, enabling greater parallelization during training.",
    },
    {
        "question": "What is the computational advantage of self-attention?",
        "reference": "Self-attention can process all positions in parallel, making it more parallelizable than recurrent approaches.",
    },
    {
        "question": "What are the main contributions of the paper?",
        "reference": "The paper introduces an attention-based Transformer architecture that removes recurrence and enables greater parallelization.",
    },
]
