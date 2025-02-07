import open_clip
from hqq.core.quantize import BaseQuantizeConfig
from hqq.engine.open_clip import HQQOpenCLIP


def load_open_clip_hqq(
    model_name: str,
    pretrained: str,
    cache_dir: str = None,
    device="cpu",
    **kwargs,
):
    model, _, transform = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, cache_dir=cache_dir
    )

    model = _quantize_model(model, model_name, **kwargs)
    model = model.to(device)

    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer


def _quantize_model(model, model_name, **kwargs):
    # Quantize settings
    budget = kwargs.get("budget", None)
    if budget is not None:
        quant_config = BaseQuantizeConfig(budget=budget, mixed=True, quant_scale=True)
        quant_config["quant_metrics_file"] = kwargs.get("quant_metrics_file")
        quant_config["weight_algo"] = kwargs.get("weight_algo", None)
        quant_config["boost_stop"] = kwargs.get("boost_stop", None)
        quant_config["ablation"] = kwargs.get("ablation", None)
        quant_config["top_m_layer"] = kwargs.get("top_m_layer", None)
    else:
        b = kwargs.get("nbits", 4)
        g = kwargs.get("group_size", 64)
        quant_config = BaseQuantizeConfig(nbits=b, group_size=g)

    model = HQQOpenCLIP.wrap_model(model, model_name)
    print(f"Using quant_config: {quant_config}")
    model.quantize_model(quant_config=quant_config)
    return model
