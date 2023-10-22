import torch

class Quantize:
    def __init__(self, x_tensor:torch.Tensor, y_type:str, q_flag:str="unsigned"):
        '''
        Input Args:
            q_flag: target (quantized) domain - 'signed' or 'unsigned'
            x_tensor: input tensor
            y_type: 'int2', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'
        '''
        assert q_flag in ["signed", "unsigned"], "target flag must be 'signed' or 'unsigned'"
        self.q_flag = q_flag # "signed" or "unsigned"

        if self.q_flag == "unsigned":
            self.range_dict = {
                        'int2': (0, 3),
                        'uint8': (0, 255),
                        'uint16': (0, 65535),
                        'uint32': (0, 4294967295),
                        'uint64': (0, 18446744073709551615),
                        'float16': (-65504, 65504),
                        'float32': (-3.4028235e+38, 3.4028235e+38),
                        'float64': (-1.7976931348623157e+308, 1.7976931348623157e+308)
                        }
        elif self.q_flag == "signed":   
            self.range_dict = {
                        'int2': (-2, 1),
                        'int8': (-128, 127),
                        'int16': (-32768, 32767),
                        'int32': (-2147483648, 2147483647),
                        'int64': (-9223372036854775808, 9223372036854775807),
                        'float16': (-65504, 65504),
                        'float32': (-3.4028235e+38, 3.4028235e+38),
                        'float64': (-1.7976931348623157e+308, 1.7976931348623157e+308)
                        }

        self.tensor = x_tensor.clone()
        self.y_type = y_type
        
    
    def run(self):
        q_min, q_max = self.range_dict[self.y_type]
        x_min, x_max = torch.min(self.tensor), torch.max(self.tensor)

        # scale factor
        s = (x_max - x_min) / (q_max - q_min)

        # zero point
        if torch.round(x_min / s) == q_min:
            z = 0
        else:
            z = -torch.round(x_min / s) + q_min

        self.tensor = torch.round(self.tensor/s) + z
        self.tensor = torch.clamp(self.tensor, q_min, q_max)

        return self.tensor
