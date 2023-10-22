import torch

class Quantize:
    def __init__(self, x_tensor:torch.Tensor, y_type:str, q_flag:str="unsigned"):
        '''
        Features:
            Quantization 
                from float to int
            Dequantization
                from int to float
        Input Args:
            q_flag: target (quantized) domain - 'signed' or 'unsigned'
            x_tensor: input tensor
            y_type: 'int2', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'
        '''
        assert q_flag in ["signed", "unsigned"], "linear mapping target flag must be 'signed' or 'unsigned'"
        self.q_flag = q_flag # "signed" or "unsigned"

        if self.q_flag == "unsigned":
            self.range_dict = {
                        'int2': (0, 3),
                        'uint8': (0, 255),
                        'uint16': (0, 65535),
                        'uint32': (0, 4294967295),
                        'uint64': (0, 18446744073709551615),
                        }
        elif self.q_flag == "signed":   
            self.range_dict = {
                        'int2': (-2, 1),
                        'int8': (-128, 127),
                        'int16': (-32768, 32767),
                        'int32': (-2147483648, 2147483647),
                        'int64': (-9223372036854775808, 9223372036854775807),
                        }

        self.tensor = x_tensor # input tensor
        self.y_type = y_type # target type
        
        self.q_min, self.q_max = self.range_dict[self.y_type]
        self.x_min, self.x_max = torch.min(self.tensor), torch.max(self.tensor)
    
    def scale_and_zero_point(self, x_min:float, x_max:float, q_min, q_max):
        # scale factor
        s = (x_max - x_min) / (q_max - q_min)

        # zero point
        if torch.round(x_min / s) == q_min:
            z = 0
        else:
            z = -torch.round(x_min / s) + q_min
        
        return s, z
    
    def linear_mapping(self):
        '''
        Quantization
        '''
        self.s, self.z = self.scale_and_zero_point(self.x_min, self.x_max, self.q_min, self.q_max)
        tensor = torch.round(self.tensor/self.s + self.z)
        tensor = torch.clamp(tensor, self.q_min, self.q_max)
        return tensor # quantized tensor

    def inverse_mapping(self):
        '''
        Dequantization
        '''
        tensor = self.linear_mapping()
        tensor = (tensor - self.z) * self.s
        return tensor # dequantized tensor
    
    def error(self):
        '''
        Quantization Error
        '''
        tensor = self.inverse_mapping()
        tensor = torch.abs(self.tensor - tensor)
        return tensor
