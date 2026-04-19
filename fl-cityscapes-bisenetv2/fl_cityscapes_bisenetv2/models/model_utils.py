from lib.models.bisenetv2 import BiSeNetV2
from fl_cityscapes_bisenetv2.models.deeplabv3p import create_deeplabv3p


def get_model(num_classes: int, model_name: str):
    if model_name == "BiSeNetV2":
        print(f"[INFO] Loading BiSeNetV2 model.")
        return BiSeNetV2(num_classes)
    elif model_name == "DeepLabV3PlusLarge":
        print(f"[INFO] Loading DeepLabV3PlusLarge model.")
        return create_deeplabv3p(num_classes, encoder_variation="large")
    elif model_name == "DeepLabV3PlusSmall":
        print(f"[INFO] Loading DeepLabV3PlusSmall model.")
        return create_deeplabv3p(num_classes, encoder_variation="small")
    else:
        raise ValueError(
            f"Invalid model_name: {model_name}. Must be 'BiSeNetV2', 'DeepLabV3PlusLarge', or 'DeepLabV3PlusSmall'."
        )
