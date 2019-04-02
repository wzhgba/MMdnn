import onnx
from onnx import helper, shape_inference, optimizer
import numpy as np
import math


def optimize_onnx_model(onnx_model):
    try:
        onnx_model = optimizer.optimize(onnx_model, passes=["nop", "eliminate_identity", "eliminate_nop_transpose", "eliminate_nop_pad",
                                                            "eliminate_unused_initializer", "fuse_consecutive_squeezes", "fuse_consecutive_transposes", "fuse_add_bias_into_conv", "fuse_transpose_into_gemm"])
    except:
        pass
    return onnx_model


def move_all_constant_node_into_initializer(onnx_model):
    nodes_to_be_copied = []
    for node in onnx_model.graph.node:
        if not node.op_type == "Constant":
            nodes_to_be_copied.append(node)
        else:
            data_list = []
            tensor_dims = list(node.attribute[0].t.dims)
            if node.attribute[0].t.data_type == onnx.TensorProto.INT32:
                data_list = node.attribute[0].t.int32_data
            elif node.attribute[0].t.data_type == onnx.TensorProto.FLOAT:
                data_list = node.attribute[0].t.float_data
            elif node.attribute[0].t.data_type == onnx.TensorProto.INT64:
                data_list = node.attribute[0].t.int64_data
            else:
                raise ValueError("data_type not supported!!!!!!")
            data_np_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[node.attribute[0].t.data_type]

            data_np = np.array(data_list, data_np_type).reshape(tensor_dims)
            new_tensor = helper.make_tensor(name=node.output[0],
                                            data_type=node.attribute[0].t.data_type,
                                            dims=data_np.shape,
                                            vals=data_np.flatten().tobytes(), raw=True)
            new_tensor_value_info = helper.make_tensor_value_info(name=node.output[0],
                                                                  elem_type=node.attribute[0].t.data_type,
                                                                  shape=data_np.shape)
            onnx_model.graph.initializer.extend([new_tensor])
            onnx_model.graph.input.extend([new_tensor_value_info])
    if nodes_to_be_copied:
        while len(onnx_model.graph.node) > 0:
            del onnx_model.graph.node[0]
        onnx_model.graph.node.extend(nodes_to_be_copied)
    return onnx_model


def fuse_bn_into_conv(onnx_model):
    whose_input = {}
    whose_output = {}
    for node in onnx_model.graph.node:
        for input_tensor in node.input:
            if input_tensor in whose_input.keys():
                whose_input[input_tensor].append(node)
            else:
                whose_input[input_tensor] = [node,]
        for output_tensor in node.output:
            if output_tensor in whose_input.keys():
                whose_output[output_tensor] .append(node)
            else:
                whose_output[output_tensor] = [node,]

    current_node = 0
    initializers_to_be_deleted = []
    initializers_name_to_id_dict = {}
    for i, ini in enumerate(onnx_model.graph.initializer):
        initializers_name_to_id_dict[ini.name] = i

    while current_node < len(onnx_model.graph.node):
        node = onnx_model.graph.node[current_node]
        if not node.op_type == "BatchNormalization" or len(whose_input[node.input[0]]) > 1 or not whose_output[node.input[0]][0].op_type == "Conv" :
            current_node += 1
            continue
		#bn should has 5 inputs and all inputs should not be empty
        elif len(node.input) < 5 or onnx_model.graph.initializer[initializers_name_to_id_dict[node.input[3]]].dims[0] == 0 or onnx_model.graph.initializer[initializers_name_to_id_dict[node.input[4]]].dims[0] == 0:
            current_node += 1
            continue
        else:
            conv_node = whose_output[node.input[0]][0]
            eps = 1e-5
            for attr in node.attribute:
                if attr.name == "epsilon":
                    eps = attr.f

            conv_w = np.frombuffer(onnx_model.graph.initializer[initializers_name_to_id_dict[conv_node.input[1]]].raw_data,
                                   onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_model.graph.initializer[initializers_name_to_id_dict[conv_node.input[1]]].data_type])
            bn_s = np.frombuffer(onnx_model.graph.initializer[initializers_name_to_id_dict[node.input[1]]].raw_data,
                                 onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_model.graph.initializer[initializers_name_to_id_dict[node.input[1]]].data_type])
            bn_b = np.frombuffer(onnx_model.graph.initializer[initializers_name_to_id_dict[node.input[2]]].raw_data,
                                 onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_model.graph.initializer[initializers_name_to_id_dict[node.input[1]]].data_type])

            #conv may has no bias, make empty array here to replace then
            channel = onnx_model.graph.initializer[initializers_name_to_id_dict[node.input[2]]].dims[0]
            conv_b = np.frombuffer(onnx_model.graph.initializer[initializers_name_to_id_dict[conv_node.input[2]]].raw_data,
                                   onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_model.graph.initializer[initializers_name_to_id_dict[conv_node.input[2]]].data_type]) if len(conv_node.input) > 2 else np.zeros([channel], onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_model.graph.initializer[initializers_name_to_id_dict[conv_node.input[1]]].data_type])
            bn_m = np.frombuffer(onnx_model.graph.initializer[initializers_name_to_id_dict[node.input[3]]].raw_data,
                                 onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_model.graph.initializer[initializers_name_to_id_dict[node.input[3]]].data_type])
            bn_v = np.frombuffer(onnx_model.graph.initializer[initializers_name_to_id_dict[node.input[4]]].raw_data,
                                 onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_model.graph.initializer[initializers_name_to_id_dict[node.input[4]]].data_type])

            position = 0
            fuse_conv_w = np.ndarray(channel * (conv_w.shape[0] // channel),onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_model.graph.initializer[initializers_name_to_id_dict[conv_node.input[1]]].data_type])
            fuse_conv_b = np.ndarray(channel,onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_model.graph.initializer[initializers_name_to_id_dict[conv_node.input[2]]].data_type]) if len(conv_node.input) > 2 else np.zeros([channel], onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_model.graph.initializer[initializers_name_to_id_dict[conv_node.input[1]]].data_type])
            for i in range(0, channel):
                s = bn_s[i] / math.sqrt(bn_v[i] + eps)
                b = bn_b[i] - bn_m[i] * s
                for j in range(0, conv_w.shape[0] // channel):
                    fuse_conv_w[position] = conv_w[position] * s
                    position += 1
                fuse_conv_b[i] = conv_b[i] * s + b

            onnx_model.graph.initializer[initializers_name_to_id_dict[conv_node.input[1]]
                                         ].raw_data = fuse_conv_w.tobytes()
            if len(conv_node.input) > 2:
                onnx_model.graph.initializer[initializers_name_to_id_dict[conv_node.input[2]]].raw_data = fuse_conv_b.tobytes(
                )
            else:
                tensor_name = conv_node.input[1] + "-polish-bias-added"
                new_tensor = helper.make_tensor(name=tensor_name,
                                                data_type=onnx_model.graph.initializer[
                                                    initializers_name_to_id_dict[conv_node.input[1]]].data_type,
                                                dims=[channel],
                                                vals=fuse_conv_b.tobytes(), raw=True)
                new_tensor_value_info = helper.make_tensor_value_info(name=tensor_name,
                                                                      elem_type=onnx_model.graph.initializer[
                                                                          initializers_name_to_id_dict[conv_node.input[1]]].data_type,
                                                                      shape=[channel])
                onnx_model.graph.initializer.extend([new_tensor])
                onnx_model.graph.input.extend([new_tensor_value_info])
                conv_node.input.extend([tensor_name])

            conv_node.output[0] = node.output[0]
            initializers_to_be_deleted.extend(node.input)
            del onnx_model.graph.node[current_node]

    current_initializer = 0
    while current_initializer < len(onnx_model.graph.initializer):
        if onnx_model.graph.initializer[current_initializer].name in initializers_to_be_deleted:
            del onnx_model.graph.initializer[current_initializer]
        else:
            current_initializer += 1

    current_input = 0
    while current_input < len(onnx_model.graph.initializer):
        if onnx_model.graph.input[current_input].name in initializers_to_be_deleted:
            del onnx_model.graph.input[current_input]
        else:
            current_input += 1
    return onnx_model


def remove_dropout(onnx_model):
    current_node = 0
    while current_node < len(onnx_model.graph.node):
        if onnx_model.graph.node[current_node].op_type == "Dropout":
            for front_node in onnx_model.graph.node:
                if front_node.output[0] == onnx_model.graph.node[current_node].input[0]:
                    front_node.output[0] = onnx_model.graph.node[current_node].output[0]
                    break
            del onnx_model.graph.node[current_node]
        else:
            current_node += 1
    return onnx_model

def find_mul_add(onnx_model):
    whose_input = {}
    whose_output = {}
    for node in onnx_model.graph.node:
        for input_tensor in node.input:
            if input_tensor not in whose_input.keys():
                whose_input[input_tensor] = [node,]
            else:
                whose_input[input_tensor].append(node)
        for output_tensor in node.output:
            if output_tensor not in whose_output.keys():
                whose_output[output_tensor] = [node,]
            else:
                whose_output[output_tensor].append(node)
    
    for name,item in whose_output.items():
        if len(item) == 1 and item[0].op_type == "Mul" and name in whose_input.keys() and len(whose_input[name]) == 1 and whose_input[name][0].op_type == "Add":
            mul_dimension = 0
            check_channel_wise = True
            for ini in onnx_model.graph.initializer:
                if ini.name == item[0].input[1]:
                    mul_dimension = ini.dims[0]
                    if len(ini.dims) != 1 or ini.dims[0] == 1:
                        check_channel_wise = False
            for ini in onnx_model.graph.initializer:
                if ini.name == whose_input[name][0].input[1]:
                    if len(ini.dims) != 1 or ini.dims[0] == 1 or ini.dims[0] != mul_dimension:
                        check_channel_wise = False
            if check_channel_wise:
                return True,item[0],whose_input[name][0]
    
    return False,None,None

def fuse_mul_add_into_conv(onnx_model):
    while True:
        result, mul_node, add_node = find_mul_add(onnx_model)
        if not result:
            return onnx_model
        
        mul_node.op_type = "Conv"
        mul_node.input.extend([add_node.input[1]])
        channel = -1
        for ini in onnx_model.graph.initializer:
            if ini.name == mul_node.input[1]:
                ini.dims.extend([1,1,1])
                channel = ini.dims[0]
        new_node = helper.make_node("Conv",["temp","temp","temp"],["temp"],kernel_shape=[1,1],pads=[0,0,0,0],strides=[1,1],group=channel)
        mul_node.attribute.extend(new_node.attribute)
        mul_node.output[0] = add_node.output[0]
        for i in range(0,len(onnx_model.graph.node)):
            if onnx_model.graph.node[i] == add_node:
                del onnx_model.graph.node[i]
                break

def find_fuseable_bn(onnx_model):
    for node in onnx_model.graph.node:
        if node.op_type == "BatchNormalization":
            channel = -1
            check_bn = True
            for ini in onnx_model.graph.initializer:
                if ini.name in node.input:
                    if channel != -1 and channel != ini.dims[0]:
                        check_bn = False
                    channel = ini.dims[0]
            if not check_bn or channel == -1:
                continue
            
            return True, node, channel
    return False, None, None

def change_bn_into_depthwise_conv(onnx_model):
    while True:
        result, bn_node, channel = find_fuseable_bn(onnx_model)
        if not result:
            return onnx_model
        

        for ini in onnx_model.graph.initializer:
            if ini.name in bn_node.input:
                pick_ini = ini
                break
        scale_np = np.zeros([channel],onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[pick_ini.data_type])
        bias_np = np.zeros([channel],onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[pick_ini.data_type])
        mean_np = np.zeros([channel],onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[pick_ini.data_type])
        var_np = np.zeros([channel],onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[pick_ini.data_type])

        merged_scale_np = np.zeros([channel],onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[pick_ini.data_type])
        merged_bias_np = np.zeros([channel],onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[pick_ini.data_type])
        for ini in onnx_model.graph.initializer:
            if ini.name == bn_node.input[1]:
                scale_np = np.frombuffer(ini.raw_data,onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[ini.data_type])
            elif ini.name == bn_node.input[2]:
                bias_np = np.frombuffer(ini.raw_data,onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[ini.data_type])
            elif ini.name == bn_node.input[3]:
                mean_np = np.frombuffer(ini.raw_data,onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[ini.data_type])
            elif ini.name == bn_node.input[4]:
                var_np = np.frombuffer(ini.raw_data,onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[ini.data_type])
        
        epsilon = 1e-5
        for attr in bn_node.attribute:
            if attr.name == "epsilon":
                epsilon = attr.f

        for i in range(0,channel):
            var = math.sqrt(var_np[i]+epsilon)
            merged_scale_np[i] = scale_np[i] / var
            merged_bias_np[i] = bias_np[i] - merged_scale_np[i] * mean_np[i]
        
        bn_node.op_type = "Conv"
        new_node = helper.make_node("Conv",["temp","temp","temp"],["temp"],kernel_shape=[1,1],pads=[0,0,0,0],strides=[1,1],group=channel)
        while len(bn_node.attribute) > 0:
            del bn_node.attribute[0]
        bn_node.attribute.extend(new_node.attribute)

        current_ini = 0

        while current_ini < len(onnx_model.graph.initializer):
            if onnx_model.graph.initializer[current_ini].name == bn_node.input[1]:
                onnx_model.graph.initializer[current_ini].dims.extend([1,1,1])
                onnx_model.graph.initializer[current_ini].raw_data = merged_scale_np.tobytes()
                current_ini+=1
            elif onnx_model.graph.initializer[current_ini].name == bn_node.input[2]:
                onnx_model.graph.initializer[current_ini].raw_data = merged_bias_np.tobytes()
                current_ini+=1
            elif onnx_model.graph.initializer[current_ini].name == bn_node.input[3]:
                del onnx_model.graph.initializer[current_ini]
            elif onnx_model.graph.initializer[current_ini].name == bn_node.input[4]:
                del onnx_model.graph.initializer[current_ini]
            else:
                current_ini+=1

        current_input = 0
        while current_input < len(onnx_model.graph.input):
            if onnx_model.graph.input[current_input].name == bn_node.input[3]:
                del onnx_model.graph.input[current_input]
            elif onnx_model.graph.input[current_input].name == bn_node.input[4]:
                del onnx_model.graph.input[current_input]
            else:
                current_input+=1
        del bn_node.input[3]
        del bn_node.input[3]

            

def clean_unused_initializers(onnx_model):
    tensors_appeared_list = []
    for node in onnx_model.graph.node:
        for input_tensor in node.input:
            if not input_tensor in tensors_appeared_list:
                tensors_appeared_list.append(input_tensor)
        for output_tensor in node.output:
            if not output_tensor in tensors_appeared_list:
                tensors_appeared_list.append(output_tensor)
    current_initializer = 0
    while current_initializer < len(onnx_model.graph.initializer):
        if onnx_model.graph.initializer[current_initializer].name in tensors_appeared_list:
            current_initializer += 1
        else:
            del onnx_model.graph.initializer[current_initializer]
    current_input = 0
    while current_input < len(onnx_model.graph.input):
        if onnx_model.graph.input[current_input].name in tensors_appeared_list:
            current_input += 1
        else:
            del onnx_model.graph.input[current_input]
    return onnx_model


def last_shape_inference(onnx_model):
    while len(onnx_model.graph.value_info) > 0:
        del onnx_model.graph.value_info[0]
    return shape_inference.infer_shapes(onnx_model)

def onnx_polish(onnx_model):
    onnx_model = optimize_onnx_model(onnx_model)
    onnx_model = move_all_constant_node_into_initializer(onnx_model)
    onnx_model = fuse_bn_into_conv(onnx_model)
    onnx_model = remove_dropout(onnx_model)
    onnx_model = fuse_mul_add_into_conv(onnx_model)
    onnx_model = change_bn_into_depthwise_conv(onnx_model)
    onnx_model = clean_unused_initializers(onnx_model)
    return onnx_model
