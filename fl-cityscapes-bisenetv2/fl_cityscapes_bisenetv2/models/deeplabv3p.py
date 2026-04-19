import segmentation_models_pytorch as smp


def create_deeplabv3p(num_classes: int, encoder_variation: str) -> smp.DeepLabV3Plus:
    """Create a DeepLabV3+ model with a MobileNetV3 backbone."""
    if encoder_variation not in ["large", "small"]:
        raise ValueError(
            f"Invalid encoder_variation: {encoder_variation}. Must be 'large' or 'small'."
        )
    encoder_name = (
        "tu-mobilenetv3_large_100"
        if encoder_variation == "large"
        else "tu-mobilenetv3_small_100"
    )
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name, encoder_weights="imagenet", classes=num_classes
    )

    return model


if __name__ == "__main__":
    # Example usage
    num_classes = 19
    encoder_variation = "small"
    model = create_deeplabv3p(num_classes, encoder_variation)
    print(model)
