shared_model_task_cache = {
    "last_task": None,
    "current_task": None,
    "models": {},
}

def get_log_posterior_from_last_task(input_shaped_tensor, active_model):
    last_task = shared_model_task_cache["last_task"]
    lvp = 0.0
    if last_task is not None:
        last_model = shared_model_task_cache["models"][shared_model_task_cache["last_task"]]
        last_model(input_shaped_tensor, sample=False)
        for name in shared_model_task_cache["modules_names_without_cls"]:
            n = name.split('.')
            if len(n) == 1:
                m = last_model._modules[n[0]]
            elif len(n) == 3:
                m = last_model._modules[n[0]]._modules[n[1]]._modules[n[2]]
            elif len(n) == 4:
                m = last_model._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]
            weight = active_model.state_dict()[f"{name}.weight_mu"]
            bias = active_model.state_dict()[f"{name}.bias_mu"]
            lp_, lvp_ = m.calculate_logs_on_external_weight_and_bias(weight, bias)
            lvp += lvp_
        return lvp, True
    return 0, False

def update_last_task(current_task):
    args = shared_model_task_cache["args"]
    if args.experiment=='mnist2' or args.experiment=='pmnist' or args.experiment == 'mnist5':
        from networks import mlp_ucb as network
    else:
        from networks import resnet_ucb as network

    shared_model_task_cache["last_task"] = shared_model_task_cache["current_task"]
    if shared_model_task_cache["last_task"] is not None:
        model_ = network.Net(args).to(args.device) 
        model_.load_state_dict(shared_model_task_cache["models"][shared_model_task_cache["last_task"]])
        model_.eval()
        shared_model_task_cache["models"][shared_model_task_cache["last_task"]] = model_
    shared_model_task_cache["current_task"] = current_task