import copy
import os
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2


def create_prototxt(model, src_prototxt):
    prototxt_file = open (src_prototxt, "w")
    model_copied = copy.deepcopy(model)
    for layer in model_copied.layer:
        layer.ClearField("blobs")
    prototxt_file.write(str(model_copied))
    prototxt_file.close()

def add_lost_scale_after_bn(caffemodel):
    src_layers = caffemodel.layer
    dst_layers = caffe_pb2.NetParameter().layer
    replace_name_map = dict()
    # Rename layer input if using bn_output
    for index, layer in enumerate(src_layers):
        for bottom_index, name in enumerate(layer.bottom):
            if name in replace_name_map.keys():
                layer.bottom.remove(name)
                layer.bottom.insert(bottom_index, replace_name_map[name])
        for top_index, name in enumerate(layer.top):
            if name in replace_name_map.keys():
                layer.top.remove(name)
                layer.top.insert(top_index, replace_name_map[name])
        dst_layers.extend([layer])
        if layer.type == 'BatchNorm' and (index + 1 >= len(src_layers) or src_layers[index + 1].type != "Scale"):
            # BN (bn_input_name -> scale_input_name) 
            # Scale (scale_input_name -> scale_output_name)
            bn_origin_output_name = layer.top[0]
            scale_input_name = layer.top[0] + "_scale_in"
            scale_output_name = layer.top[0] + "_scale_out"
            # Rename BN output 
            dst_layers[-1].top[0] = scale_input_name
            # update rename dict
            replace_name_map[bn_origin_output_name] = scale_output_name
            # Create scale layer
            scale_layer = caffe_pb2.LayerParameter()
            scale_layer.name = layer.name + "_scale"
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
    for index in range(0, len(caffemodel.layer)):
        caffemodel.layer.pop()
    caffemodel.layer.extend(dst_layers)

def caffe_polish(src_model_file, dst_model_file, src_prototxt = None, dst_prototxt = None):
    tmp_model_file = src_model_file
    
    if src_prototxt != None and dst_prototxt != None:
        # Convert caffemodel + prototxt -> temp caffemodel
        tmp_model_file = "temp_" + src_model_file
        net = caffe.Net(src_prototxt, src_model_file, caffe.TEST)
        net.save(tmp_model_file)

    caffe_model = caffe_pb2.NetParameter()
    file = open(tmp_model_file, "rb")
    caffe_model.ParseFromString(file.read())
    file.close()

    add_lost_scale_after_bn(caffe_model)

    file = open(dst_model_file, "wb")
    file.write(caffe_model.SerializeToString())
    file.close()
    
    if src_prototxt != None and dst_prototxt != None:
        if tmp_model_file != None and os.path.exists(tmp_model_file):
            os.remove(tmp_model_file)
        create_prototxt(caffe_model, dst_prototxt)
