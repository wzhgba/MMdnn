import copy
import os
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2


def create_prototxt(model, src_prototxt):
    prototxt_file = open (src_prototxt, 'w')
    model_copied = copy.deepcopy(model)
    for layer in model_copied.layer:
        layer.ClearField('blobs')
    prototxt_file.write(str(model_copied))
    prototxt_file.close()


def rename_top_and_bottom(replace_name_map, layer):
    for bottom_index, name in enumerate(layer.bottom):
        if name in replace_name_map.keys():
            layer.bottom.remove(name)
            layer.bottom.insert(bottom_index, replace_name_map[name])
    for top_index, name in enumerate(layer.top):
        if name in replace_name_map.keys():
            layer.top.remove(name)
            layer.top.insert(top_index, replace_name_map[name])

def add_lost_scale_after_bn(caffe_model):
    src_layers = caffe_model.layer
    dst_layers = caffe_pb2.NetParameter().layer
    replace_name_map = dict()
    # Rename layer input if using bn_output
    for index, layer in enumerate(src_layers):
        rename_top_and_bottom(replace_name_map, layer)
        dst_layers.extend([layer])
        if layer.type == 'BatchNorm' and (index + 1 >= len(src_layers) or src_layers[index + 1].type != "Scale"):
            print('Merge bn + scale in layer ' + str(layer.name))
            # BN (bn_input_name -> scale_input_name) 
            # Scale (scale_input_name -> scale_output_name)
            bn_origin_output_name = layer.top[0]
            scale_input_name = layer.top[0] + '_scale_in'
            scale_output_name = layer.top[0] + '_scale_out'
            # Rename BN output 
            dst_layers[-1].top[0] = scale_input_name
            # update rename dict
            replace_name_map[bn_origin_output_name] = scale_output_name
            # Create scale layer
            scale_layer = caffe_pb2.LayerParameter()
            scale_layer.name = layer.name + '_scale'
            scale_layer.type = u'Scale'
            scale_layer.bottom.append(scale_input_name)
            scale_layer.top.append(scale_output_name)
            # Add scale and bias blob in scale
            scale_blob = scale_layer.blobs.add()
            scale_blob.shape.dim.append(layer.blobs[0].shape.dim[0])
            for i in range(layer.blobs[0].shape.dim[0]):
                scale_blob.data.append(1)
            bias_blob = scale_layer.blobs.add()
            bias_blob.shape.dim.append(layer.blobs[0].shape.dim[0])
            for i in range(layer.blobs[0].shape.dim[0]):
                bias_blob.data.append(0)
            scale_layer.scale_param.bias_term = True
            # Add scale layer
            dst_layers.extend([scale_layer])
    for index in range(0, len(caffe_model.layer)):
        caffe_model.layer.pop()
    caffe_model.layer.extend(dst_layers)

def split_bnmsra_into_bn_and_scale(caffe_model):
    src_layers = caffe_model.layer
    dst_layers = caffe_pb2.NetParameter().layer
    replace_name_map = dict()
    # Rename layer input if using bn_output
    for layer in src_layers:
        rename_top_and_bottom(replace_name_map, layer)
        dst_layers.extend([layer])

        if layer.type == 'BatchNormMSRA':
            # BNMSRA (bnmsra_input_name -> scale_input_name) 
            # Scale (scale_input_name -> scale_output_name)
            bnmsra_origin_output_name = layer.top[0]
            scale_input_name = layer.top[0] + '_scale_in'
            scale_output_name = layer.top[0] + '_scale_out'
            # Rename BNMSRA type and output 
            dst_layers[-1].type = u'BatchNorm'
            dst_layers[-1].top[0] = scale_input_name
            # update rename dict
            replace_name_map[bnmsra_origin_output_name] = scale_output_name
            # Create scale layer
            scale_layer = caffe_pb2.LayerParameter()
            scale_layer.name = layer.name + '_scale'
            scale_layer.type = u'Scale'
            scale_layer.bottom.append(scale_input_name)
            scale_layer.top.append(scale_output_name)
            # Add scale and bias blob in scale
            scale_blob = scale_layer.blobs.add()
            scale_blob.shape.dim.append(layer.blobs[0].shape.dim[1])
            for i in range(layer.blobs[0].shape.dim[1]):
                scale_blob.data.append(layer.blobs[0].data[i])
            bias_blob = scale_layer.blobs.add()
            bias_blob.shape.dim.append(layer.blobs[1].shape.dim[1])
            for i in range(layer.blobs[1].shape.dim[1]):
                bias_blob.data.append(layer.blobs[1].data[i])
            scale_layer.scale_param.bias_term = True
            # update BN layer
            channel = dst_layers[-1].blobs[2].shape.dim[1]
            for i in range(len(dst_layers[-1].blobs[2].shape.dim)):
                dst_layers[-1].blobs[2].shape.dim.remove(dst_layers[-1].blobs[2].shape.dim[0])
                dst_layers[-1].blobs[3].shape.dim.remove(dst_layers[-1].blobs[3].shape.dim[0])
            dst_layers[-1].blobs[2].shape.dim.append(channel)
            dst_layers[-1].blobs[3].shape.dim.append(channel)
            dst_layers[-1].blobs.remove(dst_layers[-1].blobs[0])
            dst_layers[-1].blobs.remove(dst_layers[-1].blobs[0])
            move_blob = dst_layers[-1].blobs.add()
            move_blob.shape.dim.append(1)
            move_blob.data.append(1)
            dst_layers[-1].param.remove(dst_layers[-1].param[-1])
            dst_layers[-1].ClearField("batch_norm_msra_param")
            # Add scale layer
            dst_layers.extend([scale_layer])

    for index in range(0, len(caffe_model.layer)):
        caffe_model.layer.pop()
    caffe_model.layer.extend(dst_layers)


def caffe_polish(src_model_file, dst_model_file, src_prototxt = None, dst_prototxt = None):
    tmp_model_file = None
    if src_prototxt != None and dst_prototxt != None:
        tmp_model_file = "temp_" + src_model_file
        # Convert caffe_model + prototxt -> temp caffe_model
        net = caffe.Net(src_prototxt, src_model_file, caffe.TEST)
        net.save(tmp_model_file)
        file = open(tmp_model_file, 'rb')
    else:
        file = open(src_model_file, 'rb')

    caffe_model = caffe_pb2.NetParameter()
    caffe_model.ParseFromString(file.read())
    file.close()

    add_lost_scale_after_bn(caffe_model)
    split_bnmsra_into_bn_and_scale(caffe_model)

    file = open(dst_model_file, 'wb')
    file.write(caffe_model.SerializeToString())
    file.close()
    if src_prototxt != None and dst_prototxt != None:
        if tmp_model_file != None and os.path.exists(tmp_model_file):
            os.remove(tmp_model_file)
        create_prototxt(caffe_model, dst_prototxt)
