import torch
import random
import numpy as np
import json
import os

class DishonestServerAttacker:
    """
    模拟不诚实的聚合服务器，该服务器会篡改聚合后的全局模型。
    只篡改权重参数（包含"weight"的数组），不篡改偏置参数（包含"bias"的数组）。
    篡改比例基于所有权重参数数组元素的总数。
    """

    @staticmethod
    def tamper_global_model(global_parameters: dict, tampering_proportion: float) -> dict:
        """
        通过随机更改一部分权重参数值来篡改全局模型参数。

        Args:
            global_parameters (dict): 全局模型的参数字典，其值应为 torch.Tensor。
            tampering_proportion (float): 要篡改的权重参数的比例 (例如，0.1 表示 10%)。
                                          有效范围是 0.0 到 1.0。

        Returns:
            dict: 被篡改后的全局模型的参数字典。
        """
        if not (0.0 <= tampering_proportion <= 1.0):
            raise ValueError("篡改比例必须在 0.0 和 1.0 之间。")

        # 创建一个新的字典来存储篡改后的参数，并对Tensor进行深拷贝
        tampered_parameters = {}
        for key, value in global_parameters.items():
            if isinstance(value, torch.Tensor):
                tampered_parameters[key] = value.clone()
            else:
                # 保留非张量类型的参数（如果有的话）
                tampered_parameters[key] = value 

        # 收集所有权重张量的信息
        weight_elements_info = []  # 存储 (key, original_shape, flat_tensor)
        total_weight_elements = 0
        
        for key, value in tampered_parameters.items(): # 遍历克隆后的参数
            if "weight" in key and isinstance(value, torch.Tensor):
                if value.numel() == 0: # 跳过空张量
                    continue
                original_shape = value.shape
                flat_tensor = value.flatten()
                weight_elements_info.append({'key': key, 'original_shape': original_shape, 'flat_tensor': flat_tensor, 'device': value.device, 'dtype': value.dtype})
                total_weight_elements += flat_tensor.numel()
        
        if not weight_elements_info:
            print("警告：在模型参数中没有找到名称包含 'weight' 的张量，或这些张量为空。")
            return tampered_parameters # 返回未修改的克隆参数
            
        # 计算需要篡改的元素总数
        num_total_to_tamper = int(total_weight_elements * tampering_proportion)
        
        if num_total_to_tamper == 0:
            if tampering_proportion > 0 and total_weight_elements > 0:
                print(f"信息：计算得到的需篡改元素数量为0 (总权重元素: {total_weight_elements}, 比例: {tampering_proportion})。未执行篡改。")
            elif tampering_proportion == 0:
                print("信息：篡改比例为0，未执行篡改。")
            # 如果 total_weight_elements 为 0，则在上面已经打印过警告并返回了
            return tampered_parameters

        # 创建一个包含所有权重参数元素全局索引的列表
        # 每个元素是 (weight_elements_info中的索引, 在对应flat_tensor中的索引)
        all_indices_global = []
        for i, info in enumerate(weight_elements_info):
            for j in range(info['flat_tensor'].numel()):
                all_indices_global.append((i, j))
        
        # 确保篡改数量不超过实际元素总数
        num_total_to_tamper = min(num_total_to_tamper, len(all_indices_global))
        if num_total_to_tamper == 0 : # 再次检查，如果min操作后为0
            print(f"信息：调整后需篡改元素数量为0。未执行篡改。")
            return tampered_parameters


        # 从全局索引列表中随机选择要篡改的元素
        indices_to_tamper_global = random.sample(all_indices_global, num_total_to_tamper)
        
        tampered_count = 0
        for info_list_idx, index_within_flat_tensor in indices_to_tamper_global:
            info = weight_elements_info[info_list_idx]
            # 使用与原始张量相同设备和类型的标准正态分布随机值
            noise_value = torch.randn(1, device=info['device'], dtype=info['dtype']).item()
            info['flat_tensor'][index_within_flat_tensor] = noise_value
            tampered_count +=1
        
        if tampered_count > 0:
            print(f"信息：成功篡改 {tampered_count} 个权重参数元素。")

        # 将修改后的展平张量reshape回原始形状，并更新tampered_parameters字典
        for info in weight_elements_info:
            tampered_parameters[info['key']] = info['flat_tensor'].reshape(info['original_shape'])
        
        return tampered_parameters

def load_global_parameters(file_path):
    """加载全局模型参数"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    with open(file_path, 'r') as f:
        parameters = json.load(f)
    return parameters

def save_parameters(parameters, file_path):
    """保存参数到文件"""
    with open(file_path, 'w') as f:
        json.dump(parameters, f, indent=4)

if __name__ == '__main__':
    # 设置文件路径
    original_model_path = "./model_parameters/test/global_parameters.pt"
    tampered_model_path = "./model_parameters/test"
    
    try:
        # 加载原始全局模型参数
        print("正在加载原始全局模型参数...")
        original_parameters = load_global_parameters(original_model_path)
        
        # 设置篡改比例
        proportion_to_tamper = 0.2  # 篡改20%的权重参数
        
        print(f"\n开始篡改全局模型，篡改比例: {proportion_to_tamper*100:.0f}%...")
        
        # 执行篡改
        tampered_parameters = DishonestServerAttacker.tamper_global_model(
            original_parameters, 
            proportion_to_tamper
        )
        
        # 保存被篡改的参数
        save_parameters(tampered_parameters, tampered_model_path)
        
        # 统计篡改结果
        print("\n篡改结果统计:")
        changed_count_total = 0
        total_params_total = 0
        for k, v_tampered in tampered_parameters.items():
            v_original = original_parameters[k]
            if "weight" in k:
                changed_elements = sum(1 for a, b in zip(v_original, v_tampered) if a != b)
                total_elements = len(v_original)
                changed_count_total += changed_elements
                total_params_total += total_elements
                print(f"\n{k}:")
                print(f"  总元素数: {total_elements}")
                print(f"  被篡改元素数: {changed_elements}")
                print(f"  篡改比例: {changed_elements/total_elements:.2%}")
        
        if total_params_total > 0:
            actual_tampering_percentage = (changed_count_total / total_params_total) * 100
            print(f"\n实际篡改的权重参数总比例: {actual_tampering_percentage:.2f}%")
        else:
            print("\n模型中没有可篡改的权重参数。")
            
        print(f"\n篡改后的参数已保存到: {tampered_model_path}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
