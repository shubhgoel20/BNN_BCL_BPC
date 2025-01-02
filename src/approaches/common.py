shared_model_task_cache = {
    "last_task": None,
    "current_task": None,
    "models": {},
}

def get_log_posterior_from_last_task(input_shaped_tensor, modules_names_without_cls):
    last_task = shared_model_task_cache["last_task"]
    lvp = 0.0
    if last_task: 
        last_model = shared_model_task_cache["models"][shared_model_task_cache["last_task"]]
        last_model(input_shaped_tensor, sample=False, calculate_log_probs=True)
        for name in modules_names_without_cls:
            n = name.split('.')
            if len(n) == 1:
                m = last_model._modules[n[0]]
            elif len(n) == 3:
                m = last_model._modules[n[0]]._modules[n[1]]._modules[n[2]]
            elif len(n) == 4:
                m = last_model._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]
            lvp += m.log_variational_posterior
        return lvp, True
    return 0, False

def update_last_task(current_task):
    shared_model_task_cache["last_task"] = shared_model_task_cache["current_task"]
    if shared_model_task_cache["last_task"]:
        shared_model_task_cache["models"][shared_model_task_cache["last_task"]].eval()
    shared_model_task_cache["current_task"] = current_task