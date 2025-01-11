def get_module_parameters(active_model, ref_to_self):
    print(list(active_model.state_dict().keys()))
    print(list(ref_to_self._parameters.keys()))
    return None
