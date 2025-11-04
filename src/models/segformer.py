from transformers import SegformerForSemanticSegmentation


def segformer():
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    modules = [
        "query",
        "key",
        "value",
        "attention.output.dense",
        "mlp.dense.1",
        "mlp.dense.2",
    ]
    return model, modules
