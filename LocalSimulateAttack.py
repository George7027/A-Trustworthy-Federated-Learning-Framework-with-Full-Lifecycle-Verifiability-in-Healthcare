import torch

"""
此脚本提供了几种模拟联邦学习中恶意客户端攻击的方法，
这些攻击可以被外部程序（如server.py）调用来测试系统的鲁棒性。
"""

class ModelAttacker:
    """
    模型攻击器类，用于实施各种类型的模型参数攻击。
    提供多种攻击策略，可以单独使用或组合使用。
    """
    
    @staticmethod
    def scaling_attack(state_dict: dict, scale_factor: float = 5.0) -> dict:
        """
        将所有权重乘以一个常数因子。
        这可能会严重扭曲全局模型更新。
        
        参数:
            state_dict: 模型状态字典
            scale_factor: 缩放因子，默认为5.0
            
        返回:
            修改后的状态字典
        """
        attacked_dict = {}
        for key, tensor in state_dict.items():
            attacked_dict[key] = tensor * scale_factor
        return attacked_dict

    @staticmethod
    def noise_attack(state_dict: dict, epsilon: float = 0.1) -> dict:
        """
        向每个权重张量添加高斯噪声。
        epsilon控制噪声相对于张量标准差的幅度。
        
        参数:
            state_dict: 模型状态字典
            epsilon: 噪声强度，默认为0.1
            
        返回:
            修改后的状态字典
        """
        attacked_dict = {}
        for key, tensor in state_dict.items():
            noise = torch.randn_like(tensor) * epsilon * tensor.std()
            attacked_dict[key] = tensor + noise
        return attacked_dict

    @staticmethod
    def sign_flip_attack(state_dict: dict) -> dict:
        """
        翻转每个参数的符号，有效地将全局模型推向相反方向。
        
        参数:
            state_dict: 模型状态字典
            
        返回:
            修改后的状态字典
        """
        attacked_dict = {}
        for key, tensor in state_dict.items():
            attacked_dict[key] = -tensor
        return attacked_dict
    
    @staticmethod
    def combined_attack(state_dict: dict, attack_types: list, **kwargs) -> dict:
        """
        组合多种攻击方式。
        
        参数:
            state_dict: 模型状态字典
            attack_types: 攻击类型列表，可包含 'scaling', 'noise', 'sign_flip'
            **kwargs: 各攻击方法的参数，如scale_factor, epsilon等
            
        返回:
            修改后的状态字典
        """
        result = state_dict.copy()
        
        for attack_type in attack_types:
            if attack_type == 'scaling':
                scale_factor = kwargs.get('scale_factor', 5.0)
                result = ModelAttacker.scaling_attack(result, scale_factor)
            elif attack_type == 'noise':
                epsilon = kwargs.get('epsilon', 0.1)
                result = ModelAttacker.noise_attack(result, epsilon)
            elif attack_type == 'sign_flip':
                result = ModelAttacker.sign_flip_attack(result)
        
        return result


def load_state_dict(path: str) -> dict:
    """从给定文件路径加载PyTorch状态字典。"""
    return torch.load(path)


def save_state_dict(state_dict: dict, path: str) -> None:
    """将PyTorch状态字典保存到给定文件路径。"""
    torch.save(state_dict, path)


# 示例：如何使用此模块
if __name__ == '__main__':
    # 诚实本地参数的路径
    input_path = ''
    # 保存恶意修改参数的路径
    output_path = ''

    # 加载模型参数
    state_dict = load_state_dict(input_path)
    
    # 方法1：单独使用一种攻击
    attacked_dict = ModelAttacker.scaling_attack(state_dict, scale_factor=15.0)
    
    # 方法2：使用组合攻击
    # attacked_dict = ModelAttacker.combined_attack(
    #     state_dict, 
    #     attack_types=['scaling', 'noise'], 
    #     scale_factor=10.0,
    #     epsilon=0.05
    # )

    # 保存攻击后的参数
    save_state_dict(attacked_dict, output_path)
    print(f"已将恶意参数保存到 {output_path}")
