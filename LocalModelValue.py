import torch
import torch.nn.functional as F
import glob
import os
import traceback

def load_model_vector(file_path):
    """
    加载模型参数文件，并将所有参数展平为一个一维向量
    """
    state_dict = torch.load(file_path, map_location='cpu')
    vec_list = []
    for key, param in state_dict.items():
        vec_list.append(param.view(-1))
    return torch.cat(vec_list)


def normalize_vector(vec):
    """
    对向量进行归一化处理
    """
    norm = vec.norm() + 1e-8
    if torch.isnan(norm) or torch.isinf(norm):
        print("警告：向量范数计算出现NaN或Inf，使用默认值1.0")
        return vec / 1.0
    return vec / norm


def evaluate_local_model(local_parameters, global_parameters, cos_threshold=0.0, norm_threshold=1e-6):
    """
    评估本地模型参数
    
    参数:
        local_parameters: 本地模型参数字典
        global_parameters: 全局模型参数字典
        cos_threshold: 余弦相似度阈值
        norm_threshold: 更新向量模长阈值
    
    返回:
        dict: 包含评估结果的字典
    """
    try:
        # 初始化结果字典（默认结果，以防发生异常）
        result = {
            'is_free_rider': False,
            'is_malicious': False,
            'update_norm': 0.0,
            'cosine_similarity': 0.0,
            'details': {'reason': "默认值"}
        }
        
        # 将参数字典转换为向量
        local_vec_list = []
        global_vec_list = []
        
        # 确保两个字典有相同的键
        for key in local_parameters:
            if key in global_parameters:
                try:
                    local_vec_list.append(local_parameters[key].view(-1))
                    global_vec_list.append(global_parameters[key].view(-1))
                except Exception as e:
                    print(f"警告：无法展平键 '{key}' 的参数: {str(e)}")
        
        # 如果没有有效参数，返回默认结果
        if not local_vec_list or not global_vec_list:
            result['details']['reason'] = "无有效参数可比较"
            return result
            
        local_vec = torch.cat(local_vec_list)
        global_vec = torch.cat(global_vec_list)
        
        # 检查向量是否包含NaN或Inf
        if torch.isnan(local_vec).any() or torch.isinf(local_vec).any():
            print("警告：本地参数向量包含NaN或Inf")
            result['is_malicious'] = True
            result['details']['reason'] = "本地参数包含NaN或Inf值"
            return result
            
        if torch.isnan(global_vec).any() or torch.isinf(global_vec).any():
            print("警告：全局参数向量包含NaN或Inf")
            result['details']['reason'] = "全局参数包含NaN或Inf值，无法评估"
            return result
        
        # 计算更新向量
        update_vec = local_vec - global_vec
        update_norm_val = update_vec.norm().item()
        
        # 更新结果字典
        result['update_norm'] = update_norm_val
        
        # 检查是否为搭便车
        if update_norm_val < norm_threshold:
            result['is_free_rider'] = True
            result['details']['reason'] = f"更新向量模长过小: {update_norm_val:.2e}"
            return result
        
        # 计算余弦相似度
        local_norm = normalize_vector(local_vec)
        global_direction = normalize_vector(global_vec)
        
        # 安全计算余弦相似度
        try:
            cos_sim = F.cosine_similarity(local_norm, global_direction, dim=0)
            if torch.isnan(cos_sim) or torch.isinf(cos_sim):
                print("警告：余弦相似度计算结果为NaN或Inf")
                result['details']['reason'] = "余弦相似度计算异常"
                return result
                
            result['cosine_similarity'] = cos_sim.item()
            
            # 检查是否为恶意模型
            if cos_sim.item() < cos_threshold:
                result['is_malicious'] = True
                result['details']['reason'] = f"余弦相似度过低: {cos_sim.item():.4f}"
        except Exception as e:
            print(f"计算余弦相似度时出错: {str(e)}")
            result['details']['reason'] = f"余弦相似度计算错误: {str(e)}"
            
        return result
        
    except Exception as e:
        print(f"模型评估出现异常: {str(e)}")
        print(traceback.format_exc())
        # 返回默认结果
        return {
            'is_free_rider': False,
            'is_malicious': False,
            'update_norm': 0.0,
            'cosine_similarity': 0.0,
            'details': {'reason': f"评估过程出错: {str(e)}"}
        }


def batch_evaluate_local_models(global_model_path, local_pattern, cos_threshold=0.0, norm_threshold=1e-6):
    """
    批量评估多个本地模型文件
    
    参数:
        global_model_path: 全局模型文件路径
        local_pattern: 本地模型文件路径模式
        cos_threshold: 余弦相似度阈值
        norm_threshold: 更新向量模长阈值
    
    返回:
        dict: 包含所有评估结果的字典
    """
    try:
        # 加载全局模型
        global_vec = load_model_vector(global_model_path)
        
        # 获取所有本地模型文件
        local_files = glob.glob(local_pattern)
        
        results = {
            'free_riders': [],
            'malicious_models': [],
            'details': {}
        }
        
        for file in local_files:
            try:
                local_vec = load_model_vector(file)
                update_vec = local_vec - global_vec
                update_norm_val = update_vec.norm().item()
                
                if update_norm_val < norm_threshold:
                    results['free_riders'].append(file)
                    results['details'][file] = {
                        'status': 'free_rider',
                        'update_norm': update_norm_val
                    }
                else:
                    local_norm = normalize_vector(local_vec)
                    global_direction = normalize_vector(global_vec)
                    cos_sim = F.cosine_similarity(local_norm, global_direction, dim=0)
                    
                    if cos_sim.item() < cos_threshold:
                        results['malicious_models'].append(file)
                    
                    results['details'][file] = {
                        'status': 'malicious' if cos_sim.item() < cos_threshold else 'normal',
                        'cosine_similarity': cos_sim.item(),
                        'update_norm': update_norm_val
                    }
            except Exception as e:
                print(f"评估文件 {file} 时出错: {str(e)}")
                results['details'][file] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    except Exception as e:
        print(f"批量评估出现异常: {str(e)}")
        print(traceback.format_exc())
        return {
            'free_riders': [],
            'malicious_models': [],
            'details': {'error': str(e)}
        }


if __name__ == '__main__':
    # 文件路径配置
    base_path1 = r"./model_parameters/_Test_MNIST/10_clients/Step1_pt"
    base_path2 = r"./model_parameters/_Test_MNIST/10_clients/Step1_pt"
    global_model_path = os.path.join(base_path1, "local_parameters_client5.pt")
    local_pattern = os.path.join(base_path2, "local_parameters_client*.pt")
    
    # 执行批量评估
    import time
    eval_times = []
    for i in range(100):
        start = time.perf_counter()
        results = batch_evaluate_local_models(global_model_path, local_pattern)
        elapsed = time.perf_counter() - start
        eval_times.append(elapsed)
    avg_eval_time = sum(eval_times) / len(eval_times)
    print(f"All batch evaluation times: {[f'{t:.4f}' for t in eval_times]}")
    print(f"Average batch evaluation time over 100 runs: {avg_eval_time:.4f} seconds")

    # # 打印结果
    # print("\n搭便车（无贡献）模型文件列表：")
    # for f in results['free_riders']:
    #     print(f)
    
    # print("\n恶意模型文件列表（余弦相似度低于阈值）：")
    # for f in results['malicious_models']:
    #     print(f)

