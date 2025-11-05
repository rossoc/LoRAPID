from transformers import Mask2FormerForUniversalSegmentation


def mask2former():
    model_name = "facebook/mask2former-swin-small-ade-semantic"
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "background", 1: "polyp"},
        label2id={"background": 0, "polyp": 1},
        ignore_mismatched_sizes=True,
    )

    model.train()

    modules = [
        # Backbone attention layers
        "query",
        "key",
        "value",
        "attention.output.dense",
        # Backbone MLP layers
        "intermediate.dense",
        "output.dense",
        # Pixel decoder layers
        "value_proj",
        "output_proj",
        "fc1",
        "fc2",
        # Transformer decoder layers
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "in_proj_weight",
        "in_proj_bias",
        # Prediction heads
        "class_predictor",
        "mask_embedder",
        "mask_projection",
        # Other linear projections
        "projection",
        "reduction",
        "dense",
    ]
    return model, model_name, modules
