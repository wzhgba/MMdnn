import cntk

def remove_ending_training_function(model):
    extra_function_list = ['Combine', 'CrossEntropyWithSoftmax', 'ClassificationError', 'ElementTimes', 'AdditiveFullConnection', 'FeatureNormalize']
    functionList = [model.root_function]
    visited = []
    while functionList:
        function = functionList.pop(0)
        if function.op_name in extra_function_list:
            print("Remove function: " + function.name)
        else:
            return cntk.combine([function])

        for argu in function.arguments:
            if argu.is_output and argu.owner.uid not in visited:
                functionList.append(argu.owner)
                visited.append(argu.owner.uid)
    return model

def cntk_polish(src_model_file, dst_model_file):
    src_model = cntk.load_model(src_model_file)
    dst_model = remove_ending_training_function(src_model)
    dst_model.save(dst_model_file)