from transformers import SegformerForSemanticSegmentation


def segformer():
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "background", 1: "polyp"},
        label2id={"background": 0, "polyp": 1},
        ignore_mismatched_sizes=True,
    )

    model.train()

    modules = [
        "query",
        "key",
        "value",
        "attention.output.dense",
        "mlp.dense.1",
        "mlp.dense.2",
    ]
    return model, model_name, modules
