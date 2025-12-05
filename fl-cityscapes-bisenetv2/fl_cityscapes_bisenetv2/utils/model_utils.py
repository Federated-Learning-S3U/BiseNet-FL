import torch


def set_optimizer(model, lr_start, weight_decay):
    if hasattr(model, "get_params"):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = (
            model.get_params()
        )
        #  wd_val = cfg.weight_decay
        wd_val = 0
        params_list = [
            {
                "params": wd_params,
            },
            {"params": nowd_params, "weight_decay": wd_val},
            {"params": lr_mul_wd_params, "lr": lr_start * 10},
            {
                "params": lr_mul_nowd_params,
                "weight_decay": wd_val,
                "lr": lr_start * 10,
            },
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {
                "params": wd_params,
            },
            {"params": non_wd_params, "weight_decay": 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=lr_start,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    return optim
