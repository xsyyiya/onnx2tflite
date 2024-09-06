#author: bst shiyong.xiong
import re
import os
import datetime
import numpy as np
import torch
import tensorflow as tf
import flatbuffers
import copy
from tensorflow.lite.python import schema_py_generated as schema_fb
from tqdm import tqdm

import argparse

class MyParseTflite(object):
    '''
    Parameters
    -----------
        tflite_path:  str
            Path of tflite model.

        model_name: str, optional, default 'model'.
            Name of tflite model.

        save_path: str, optional, default './'.
            Path of saving data.

        ignore_ops: list, optional
            Specify which operators do not need to be exported, default [None].

            Particular attention:  When setting ignore_ops, It is necessary to ensure that the output 
        of the ignored operator is not called by subsequent operators. otherwise, Makeing sure to run 
        with default selection before setting it up arbitrarily 

        is_saving_data: bool, optional
            whether to save data, default False.
    -----------    
    '''
    def __init__(
        self, 
        tflite_path:str, 
        model_name:str = 'model', 
        save_path:str = './', 
        ignore_ops:list = [None], 
        is_saving_data=False,
    ):
        self.tflite_path = tflite_path
        self.save_path = save_path
        self.model_name = model_name
        if is_saving_data:
            self.__mkdir()
            self.per_ops_out = self.__load_npy()

        self.model_buffer = self.__load() 
        self.interpreter = tf.lite.Interpreter(model_content=self.model_buffer)
        self.interpreter.allocate_tensors()
        self.tensor_details = self.interpreter.get_tensor_details()
        self.ignore_ops = ignore_ops
        self.ops_types = []
        self.locations = self.__get_location_id()
        self.is_saving_data = is_saving_data 

    def __mkdir(self):
        if not os.path.exists(f'{self.save_path}'):
            os.mkdir(f'{self.save_path}')
        if not os.path.exists(f'{self.save_path}/{self.model_name}'):
            os.mkdir(f'{self.save_path}/{self.model_name}')  #or use os.makedirs

    def __load(self):
        with open(self.tflite_path, 'rb') as f:
            model_buffer = f.read()
        return model_buffer
    
    def __persist(self, s):
        if not os.path.exists(f'{self.save_path}/tmp'):
            os.mkdir(f'{self.save_path}/tmp')
        if not os.path.exists(f'{self.save_path}/tmp/{self.model_name}'):
            os.mkdir(f'{self.save_path}/tmp/{self.model_name}')
        file_path = f'{self.save_path}/tmp/{self.model_name}/per_ops_out_{s}.npy'
        np.save(file_path, self.per_ops_out)

    def __new_file(self, dir_path):
        listt = os.listdir(dir_path)
        if any(listt):
            listt.sort(key=lambda fn:os.path.getmtime(dir_path+'/'+fn))
            filetime = datetime.datetime.fromtimestamp(os.path.getmtime(dir_path+'/'+listt[-1]))
            filepath = os.path.join(dir_path, listt[-1])
            return filepath
        else:
            raise Exception("there is no temp files")

    def __load_npy(self):
        per_ops_out = {}
        file_path = f'{self.save_path}/tmp/{self.model_name}'
        if os.path.exists(file_path):
            npy_file_path = self.__new_file(file_path)
            per_ops_out = np.load(npy_file_path, allow_pickle=True).item()

        return per_ops_out 
    
    def __get_location_id(self):     
        data = self.__CreateDictFromFlatbuffer(self.model_buffer)
        op_codes = data['operator_codes']  #支持/注册的op
        subg = data['subgraphs'][0] #模型结构描述，具体的op构成

        locations={}
        op_ = {}
        for i, layer in enumerate(subg['operators']):
            op_idx = layer['opcode_index']
            op_code = op_codes[op_idx]['builtin_code']
            layer_name = self.__BuiltinCodeToName(op_code) 

            #统计模型中用到的算子种类
            if layer_name not in self.ops_types:
                self.ops_types.append(layer_name)

            # 设置哪些算子不用保存数据
            if i not in self.ignore_ops:
                op_['in'] = layer['inputs']
                op_['out'] = layer['outputs']
                key = f'{layer_name}_{i}'
                locations[key] = copy.deepcopy(op_)

            #因为忽视的最后一个算子的输出需要作为下一个算子的输入，所以这里单独处理，得到其输出的tensor。注意这里对最后一个被忽视算子的设置是游要求的，详见接口说明。
            if i == self.ignore_ops[-1]:
                op_['in'] = [None]
                op_['out'] = layer['outputs']
                key = f'{layer_name}_{i}'
                locations[key] = copy.deepcopy(op_)
                
        return locations

    def __BuiltinCodeToName(self, code):
        """Converts a builtin op code enum to a readable name."""
        for name, value in schema_fb.BuiltinOperator.__dict__.items():
            if value == code:
                return name
        return None

    def __CamelCaseToSnakeCase(self, camel_case_input):
        """Converts an identifier in CamelCase to snake_case."""
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_input)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    
    def __FlatbufferToDict(self, fb, preserve_as_numpy):
        if isinstance(fb, int) or isinstance(fb, float) or isinstance(fb, str):
            return fb
        elif hasattr(fb, "__dict__"):
            result = {}
            for attribute_name in dir(fb):
                attribute = fb.__getattribute__(attribute_name)
                if not callable(attribute) and attribute_name[0] != "_":
                    snake_name = self.__CamelCaseToSnakeCase(attribute_name)
                    preserve = True if attribute_name == "buffers" else preserve_as_numpy
                    result[snake_name] = self.__FlatbufferToDict(attribute, preserve)
            return result
        elif isinstance(fb, np.ndarray):
            return fb if preserve_as_numpy else fb.tolist()
        elif hasattr(fb, "__len__"):
            return [self.__FlatbufferToDict(entry, preserve_as_numpy) for entry in fb]
        else:
            return fb

    def __CreateDictFromFlatbuffer(self, buffer_data):
        model_obj = schema_fb.Model.GetRootAsModel(buffer_data, 0)
        model = schema_fb.ModelT.InitFromObj(model_obj)
        return self.__FlatbufferToDict(model, preserve_as_numpy=False)

    def __OutputsOffset(self, subgraph, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(subgraph._tab.Offset(8))
        if o != 0:
            a = subgraph._tab.Vector(o)
            return a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4)
        return 0
    
    def __buffer_change_output_tensor_to(self, model_buffer, new_tensor_i):
        
        root = schema_fb.Model.GetRootAsModel(model_buffer, 0)
        output_tensor_index_offset = self.__OutputsOffset(root.Subgraphs(0), 0)
        
        # Flatbuffer scalars are stored in little-endian.
        new_tensor_i_bytes = bytes([
        new_tensor_i & 0x000000FF, \
        (new_tensor_i & 0x0000FF00) >> 8, \
        (new_tensor_i & 0x00FF0000) >> 16, \
        (new_tensor_i & 0xFF000000) >> 24 \
        ])
        # Replace the 4 bytes corresponding to the first output tensor index
        return model_buffer[:output_tensor_index_offset] + new_tensor_i_bytes + model_buffer[output_tensor_index_offset + 4:]

    def __ch_out(self, idx, input_data:dict):
        model_buffer = self.__buffer_change_output_tensor_to(self.model_buffer, idx)
        interpreter = tf.lite.Interpreter(model_content=model_buffer) 
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        for i in range(len(input_details)):
            interpreter.set_tensor(input_details[i]['index'], input_data[f'{i}'])

            idx = input_details[i]['index']
            if type(input_data[f'{i}']) is np.ndarray:
                self.per_ops_out[f'{idx}'] = input_data[f'{i}']
            else:
                self.per_ops_out[f'{idx}'] = input_data[f'{i}'].numpy()

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data

    def inference(self, input_data:dict):
        """
        input_data: 
            Input model data, a dict that the key is location ID of op 
        """
        ops_num = len(self.locations.keys())

        count = 0
        count_all = 0
        if self.is_saving_data:
            for x in tqdm(self.locations.keys()):
                count += 1
                if not os.path.exists(f'{self.save_path}/{self.model_name}/{x}'):
                    os.mkdir(f'{self.save_path}/{self.model_name}/{x}')

                for idx in self.locations[x]['out']:
                    if f'{idx}' not in self.per_ops_out.keys():
                        output_data = self.__ch_out(idx, input_data) 
                        self.per_ops_out[f'{idx}'] = output_data

                self.__save_bin_txt(x)
                
                #每完成10%节点后保存输出数据
                if count/ops_num >= 0.1:
                    count_all += count
                    s = int(np.floor(count_all*100/ops_num))
                    count = 0
                    self.__persist(s)

            #保存所有节点输出数据
            self.__persist(100)
    
    def __savefile(self, fliename, data):    
         with open(fliename, 'wb') as f:
             f.write(data.squeeze())

    def __check_data_dim(self, data):
        if data.size > 1:
            data = data.squeeze()
        else:
            da = np.ndarray(1)
            da[0] = data
            data = da
        if data.ndim > 2:
            data = data.reshape([-1, data.shape[-1]])
        return data

    def __save_bin_txt(self, op_name):
        for key in self.locations[op_name].keys():
            for i, idx in enumerate(self.locations[op_name][key]):
                if idx is not None and idx != -1:
                    try:
                        data = self.per_ops_out[f'{idx}']
                    except:
                        # print(f'op_name: {op_name}    key : {key}    idx: {idx} ')
                        data = self.interpreter.get_tensor(idx)
                    dtype = data.dtype
                    data = self.__check_data_dim(data)

                    self.__savefile(f'{self.save_path}/{self.model_name}/{op_name}/{key}{i}_{dtype}.bin', data)
                    np.savetxt(f'{self.save_path}/{self.model_name}/{op_name}/{key}{i}_{dtype}.txt', data, fmt='%4d', delimiter=',')

def argpars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite_path', type=str, required=True, help='The path to tflite model')
    parser.add_argument('--model_name', type=str, default='model', required=False, help='Name of tflite model.')
    parser.add_argument('--save_path', type=str, default='./', required=False, help='Path of saving data.')
    parser.add_argument('--ignore_ops', type=list, default=[None], required=False, help='Specify which operators do not need to be exported, default [None].')
    parser.add_argument('--is_saving_data', type=bool, default=False, required=False, help='whether to save data, default False.')
    opt = parser.parse_args()
    return opt

def main():
    opt = argpars()
    extra = MyParseTflite(
        tflite_path = opt.tflite_path,
        model_name = opt.model_name,
        save_path = opt.save_path,
        ignore_ops = opt.ignore_ops,
        is_saving_data = opt.is_saving_data,
    )
    import pdb
    pdb.set_trace()
    # print(extra.ops_types)


if __name__ == "__main__":
    #example
    main()
    # extractor = MyParseTflite(
    #     tflite_path="/home/xsy/asr/voice_assistant_asr/model/tflite-model/decoder_int8_realdata.tflite",
    #     model_name='decoder_int8_realdata', 
    #     save_path="./params", 
    #     ignore_ops=list(np.arange(4,246)),
    #     is_saving_data = True,
    # )

    # in_data = torch.tensor([[19, 28]], dtype=torch.int64)
    # input_data = {'0': in_data}
    # extractor.inference(input_data)



