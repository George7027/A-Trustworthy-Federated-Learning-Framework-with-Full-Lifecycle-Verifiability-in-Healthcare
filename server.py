import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional
from torch import optim
from Models import LeNet5, RestNet18, LeNet1D, SimpleCNN1D, ResNet20
from clients import ClientsGroup, client
import torchvision.models as models
import torch.nn as nn
from LocalSimulateAttack import ModelAttacker
from DishonestServerAttack import DishonestServerAttacker
import sys  
import datetime  
from LocalModelValue import evaluate_local_model  

'''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--local_epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batch_size', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('--Attack', type=str, default='no', help='Whether to enable the attack simulation function(e.g. yes/no)')
parser.add_argument('--Attack_type', type=str, default='sign_flip', help='Attack type(e.g. scaling/noise/sign_flip/combined)')
parser.add_argument('--MaliNum', type=int, default=0, help='The number of malicious clients')
parser.add_argument('--Local_Model_Value', type=str, default='no', help='Whether to enable the local model evaluation function(e.g. yes/no)')
'''


# 中间参数保存路径
def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


# 模型参数保存路径
model_parameters_dir = './model_parameters'
if not os.path.exists(model_parameters_dir):
    os.makedirs(model_parameters_dir)

# 可选
def add_noise(parameters, dp, dev):
    try:
        # 检查参数是否包含NaN或Inf
        if torch.isnan(parameters).any() or torch.isinf(parameters).any():
            print(f"警告: 参数包含NaN或Inf值，不添加噪声")
            return parameters
            
        noise = None
        # 不加噪声
        if dp == 0:
            return parameters
        # 拉普拉斯噪声
        elif dp == 1:
            noise = torch.tensor(np.random.laplace(0, sigma, parameters.shape)).to(dev)
        # 高斯噪声
        else:
            noise = torch.cuda.FloatTensor(parameters.shape).normal_(0, sigma)

        return parameters.add_(noise)
    except Exception as e:
        print(f"添加噪声时出错: {str(e)}")
        return parameters  # 出错时返回原参数


# 根据配置进行模型攻击
def apply_attack(parameters, attack_type):
    """
    根据配置的攻击类型对模型参数进行攻击
    
    参数:
        parameters: 模型参数
        attack_type: 攻击类型
    返回:
        被攻击后的参数
    """
    try:
        if attack_type == "scaling":
            return ModelAttacker.scaling_attack(parameters, scale_factor=10.0)
        elif attack_type == "noise":
            return ModelAttacker.noise_attack(parameters, epsilon=0.2)
        elif attack_type == "sign_flip":
            return ModelAttacker.sign_flip_attack(parameters)
        elif attack_type == "combined":
            return ModelAttacker.combined_attack(
                parameters, 
                attack_types=['scaling', 'noise'], 
                scale_factor=5.0, 
                epsilon=0.1
            )
        else:
            # 默认不进行攻击
            return parameters
    except Exception as e:
        print(f"应用攻击时出错: {str(e)}")
        return parameters  # 出错时返回原参数


if __name__ == "__main__":

    # 定义解析器
    parser = argparse.ArgumentParser(description='FedAvg')
    parser.add_argument('-c', '--conf', dest='conf')
    arg = parser.parse_args()

    # 解析器解析json文件
    with open(arg.conf, 'r') as f:
        args = json.load(f)
    
    # 创建日志目录
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建带时间戳的日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"run_{timestamp}.txt")
    
    # 创建日志文件并重定向标准输出和标准错误
    log_file = open(log_filename, "w")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # 创建一个同时输出到控制台和文件的类
    class Logger:
        def __init__(self, file, original):
            self.file = file
            self.original = original
            
        def write(self, message):
            self.original.write(message)
            self.file.write(message)
            
        def flush(self):
            self.original.flush()
            self.file.flush()
    
    # 重定向标准输出和标准错误
    sys.stdout = Logger(log_file, original_stdout)
    sys.stderr = Logger(log_file, original_stderr)
    
    print(f"开始运行 - 时间: {timestamp}")
    print(f"配置文件: {arg.conf}")
    print(f"参数配置: {args}")

    # 创建中间参数保存目录
    test_mkdir(args['save_path'])

    # 使用gpu or cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 定义使用模型(全连接 or 简单卷积)
    net = None
    if args['model_name'] == 'lenet5':
        net = LeNet5()
        print("Model: LeNet-5")
    elif args['model_name'] == 'resnet18':
        net = RestNet18()
        print("Model: ResNet-18")
        # net = models.resnet18(pretrained=True)
        # net.fc = nn.Linear(net.fc.in_features, 10)
    elif args['model_name'] == 'lenet1d':
        net = LeNet1D()
        print("Model: LeNet-1D")
    elif args['model_name'] == 'simplecnn1d':
        net = SimpleCNN1D()
        print("Model: SimpleCNN-1D")
    elif args['model_name'] == 'resnet20':
        net = ResNet20()
        print("Model: ResNet-20")
    else:
        raise ValueError(f"Unsupported model: {args['model_name']}")

    # 检查GPU设备，如不止一个，并行计算
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     net = torch.nn.DataParallel(net)
    net = net.to(dev)

    # 定义损失函数和优化器
    loss_func = torch.nn.functional.cross_entropy
    lr = args['learning_rate']
    val_loss_min = np.Inf

    # 定义多个参与方，导入训练、测试数据集
    myClients = ClientsGroup(args['type'], args['IID'], args['num_of_clients'], dev)
    print("\nDataset:", args['type'])
    testDataLoader = myClients.test_data_loader
    trainDataLoader = myClients.train_data_loader

    # 每轮迭代的参与方个数
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    # 初始化全局参数
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    # 定义噪声的类型和幅度（可选）
    dp = args['noise']
    sigma = args['sigma']

    # 获取攻击配置
    attack_enabled = args.get('Attack', 'no').lower() == 'yes'
    attack_type = args.get('Attack_type', '')
    mali_num = int(args.get('MaliNum', 0))
    
    if attack_enabled:
        print(f"攻击已启用! 类型: {attack_type}, 恶意客户端数量: {mali_num}")
    else:
        print("攻击未启用")
        
    # 获取模型评估配置
    model_evaluation_enabled = args.get('Local_Model_Value', 'no').lower() == 'yes'
    if model_evaluation_enabled:
        print("本地模型评估已启用!")
    else:
        print("本地模型评估未启用!")

    # 获取恶意服务器攻击配置
    dishonest_server_attack_enabled = args.get('Dishonest_Server_Attack', 'no').lower() == 'yes'
    dishonest_server_attack_proportion = float(args.get('Dishonest_Server_Tampering_Proportion', 0.0))

    if dishonest_server_attack_enabled:
        print(f"恶意服务器模拟攻击已启用! 类型: 随机篡改全局模型参数，篡改比例: {dishonest_server_attack_proportion}")
    else:
        print("恶意服务器模拟攻击未启用")

    # 保存训练集accuracy和验证集accuracy
    train_acc = []
    val_acc = []
    # 保存训练集和验证集合的loss
    train_loss = []
    val_loss = []

    # 全局迭代轮次
    for i in range(args['num_comm']):
        print("..." * 30)
        print("communicate round {}".format(i + 1))

        # 为当前轮次创建一个新的子目录
        round_dir = os.path.join(model_parameters_dir, f'communicate_round_{i + 1}')
        if not os.path.exists(round_dir):
            os.makedirs(round_dir)

        # 确保 round_dir 已经被正确创建
        assert os.path.exists(round_dir), f"Directory {round_dir} was not created successfully."

        # 在opti上可对lr进行调整
        opti = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        print("learning rate: {}".format(lr))

        # # 打乱排序，确定num_in_comm个参与方
        # order = np.random.permutation(args['num_of_clients'])
        # clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        print("number_of_clients: {}".format(args['num_of_clients']))
        print("number of participating clients: {}".format(int(args['num_of_clients'] * args['cfraction'])))
        clients_in_comm = ['client{}'.format(i) for i in range(num_in_comm)]


        sum_parameters = None
        actual_clients_count = 0  # 初始化实际参与聚合的客户端计数

        # 可视化进度条对选中参与方local_epoch
        idx = 0  # 初始化idx为0
        for client in tqdm(clients_in_comm):
            # 本地梯度下降
            local_parameters = myClients.clients_set[client].localUpdate(args['local_epoch'], args['batch_size'], net,
                                                                         loss_func, opti, global_parameters)
            
            # 保存本地参数
            local_param_path = os.path.join(round_dir, f'local_parameters_{client}.pt')
            torch.save(local_parameters, local_param_path)

            # 如果启用针对本地模型的模拟攻击并且当前客户端索引小于恶意客户端数量，则进行本地模拟攻击
            if attack_enabled and idx < mali_num:
                tqdm.write(f"对客户端 {client} 进行 {attack_type} 攻击")
                local_parameters = apply_attack(local_parameters, attack_type)
                
                # 保存攻击后的参数（可选，用于分析）
                attacked_param_path = os.path.join(round_dir, f'attacked_local_parameters_{client}.pt')
                torch.save(local_parameters, attacked_param_path)
                      
            # 是否启用模型评估
            if model_evaluation_enabled:
                try:
                    # 评估本地模型
                    evaluation_result = evaluate_local_model(
                        local_parameters=local_parameters,
                        global_parameters=global_parameters,
                        cos_threshold=0.0,  # 余弦相似度阈值
                        norm_threshold=1e-6  # 更新向量模长阈值
                    )
                    
                    # 根据评估结果决定是否接受该客户端的更新
                    if evaluation_result.get('is_free_rider', False):
                        reason = evaluation_result.get('details', {}).get('reason', '未知原因')
                        tqdm.write(f"客户端 {client} 检测到搭便车行为: {reason}")
                        # 保存标记为搭便车的参数
                        free_rider_path = os.path.join(round_dir, f'local_parameters_{client}_free_rider.pt')
                        torch.save(local_parameters, free_rider_path)
                        # 跳过该客户端，不将其参数纳入聚合计算
                    elif evaluation_result.get('is_malicious', False):
                        reason = evaluation_result.get('details', {}).get('reason', '未知原因')
                        tqdm.write(f"客户端 {client} 检测到恶意行为: {reason}")
                        # 保存标记为恶意的参数
                        malicious_path = os.path.join(round_dir, f'local_parameters_{client}_malicious.pt')
                        torch.save(local_parameters, malicious_path)
                        # 跳过该客户端，不将其参数纳入聚合计算
                    else:
                        # 正常模型，输出评估结果并加入聚合
                        update_norm = evaluation_result.get('update_norm', 0.0)
                        cosine_similarity = evaluation_result.get('cosine_similarity', 0.0)
                        tqdm.write(f"客户端 {client} 正常: 更新向量模长={update_norm:.6f}, 余弦相似度={cosine_similarity:.6f}")
                        
                        try:
                            # 加入聚合
                            if sum_parameters is None:
                                sum_parameters = {}
                                for key, var in local_parameters.items():
                                    sum_parameters[key] = var.clone()
                                    sum_parameters[key] = add_noise(sum_parameters[key], dp, dev)
                            else:
                                for key in sum_parameters:
                                    if key in local_parameters:
                                        sum_parameters[key].add_(add_noise(local_parameters[key], dp, dev))
                            
                            # 增加实际参与聚合的客户端计数
                            actual_clients_count += 1
                        except Exception as e:
                            tqdm.write(f"将客户端 {client} 参数加入聚合时出错: {str(e)}")
                except Exception as e:
                    tqdm.write(f"评估客户端 {client} 时出现异常: {str(e)}")
                    # 发生异常时，不将此客户端纳入聚合
            else:
                # 模型评估未启用，所有客户端都参与聚合
                try:
                    if sum_parameters is None:
                        sum_parameters = {}
                        for key, var in local_parameters.items():
                            sum_parameters[key] = var.clone()
                            sum_parameters[key] = add_noise(sum_parameters[key], dp, dev)
                    else:
                        for key in sum_parameters:
                            if key in local_parameters:
                                sum_parameters[key].add_(add_noise(local_parameters[key], dp, dev))
                    
                    # 实际参与聚合的客户端计数
                    actual_clients_count += 1
                except Exception as e:
                    tqdm.write(f"处理客户端 {client} 参数时出错: {str(e)}")
            
            idx += 1  # 每次循环结束后将idx加1

        # 更新全局梯度参数，只有在有参与聚合的客户端时才更新
        if sum_parameters is not None and actual_clients_count > 0:
            for var in global_parameters:
                global_parameters[var] = (sum_parameters[var] / actual_clients_count)
            
            if model_evaluation_enabled:
                print(f"本轮实际参与聚合的客户端数量: {actual_clients_count}（已过滤恶意或搭便车模型）")
            else:
                print(f"本轮实际参与聚合的客户端数量: {actual_clients_count}（全部客户端均参与聚合）")
        else:
            print("---!!! 注意：本轮没有客户端参与聚合，全局模型保持不变 !!!---")

        # 保存每轮的global_parameters到当前轮次的目录
        global_param_path = os.path.join(round_dir, 'global_parameters.pt')
        torch.save(global_parameters, global_param_path)

        # >>> 开始：恶意服务器攻击模块 <<<
        if dishonest_server_attack_enabled and dishonest_server_attack_proportion > 0:
            if sum_parameters is not None and actual_clients_count > 0: # 仅当全局模型实际更新后才攻击
                print(f"---!!! 应用不诚实服务器攻击：篡改聚合后的全局模型，比例: {dishonest_server_attack_proportion*100:.0f}% !!!---")
                try:
                    global_parameters = DishonestServerAttacker.tamper_global_model(
                        global_parameters,
                        dishonest_server_attack_proportion
                    )
                    # 可选：保存被篡改的全局模型，用于分析
                    tampered_global_param_path = os.path.join(round_dir, f'tampered_global_parameters_prop{dishonest_server_attack_proportion}.pt')
                    torch.save(global_parameters, tampered_global_param_path)
                    print(f"---!!! 篡改后的全局模型已保存至: {tampered_global_param_path} !!!---")
                except Exception as e:
                    print(f"---!!! 应用不诚实服务器攻击时发生错误: {str(e)} !!!---")
            elif not (sum_parameters is not None and actual_clients_count > 0):
                 print(f"---!!! 跳过不诚实服务器攻击：本轮全局模型未更新 !!!---")
        # >>> 结束：恶意服务器攻击模块 <<<

        # 不进行计算图构建（无需反向传播）
        with torch.no_grad():
            # 满足评估的条件，用测试集进行数据评估
            if (i + 1) % args['val_freq'] == 0:
                # strict表示key、val严格重合才能执行（false不对齐部分默认初始化）
                net.load_state_dict(global_parameters, strict=True)

                # 初始化计算参数
                train_count = 0
                test_count = 0
                train_loss = 0
                val_loss = 0
                train_sum_accu = 0
                val_sum_accu = 0
                train_total_loss = 0
                val_total_loss = 0

                # 遍历每个训练数据
                for data, label in trainDataLoader:
                    # 转成gpu数据
                    data, label = data.to(dev), label.to(dev)
                    # 预测（返回结果是概率向量）
                    preds = net(data)
                    train_total_loss += torch.nn.functional.cross_entropy(preds, label,
                                                                          reduction='mean').item()
                    # 取最大概率label
                    preds = torch.argmax(preds, dim=1)
                    train_sum_accu += (preds == label).float().mean()
                    train_count += 1
                # print('train_count: {}'.format(train_count))
                print(f'train_accuracy: {100 * train_sum_accu / train_count:.2f}%')
                train_loss = train_total_loss / train_count
                print("train_loss: {}".format(train_loss))
                train_acc.append((train_sum_accu / train_count).cpu())

                # 遍历每个测试数据
                for data, label in testDataLoader:
                    # 转成gpu数据
                    data, label = data.to(dev), label.to(dev)
                    # 预测（返回结果是概率向量）
                    preds = net(data)
                    val_total_loss += torch.nn.functional.cross_entropy(preds, label,
                                                                        reduction='mean').item()
                    # 取最大概率label
                    preds = torch.argmax(preds, dim=1)
                    val_sum_accu += (preds == label).float().mean()
                    test_count += 1
                # print('test_count: {}'.format(test_count))
                print(f'val_accuracy: {100 * val_sum_accu / test_count:.2f}%')
                val_acc.append((val_sum_accu / test_count).cpu())
                val_loss = val_total_loss / test_count
                print("val_loss: {}".format(val_loss))

                # 如果验证集损失函数减小，保存最优模型
                if i > 50 and val_loss <= val_loss_min:
                    val_loss_min = val_loss
                    print("val_loss_min: {}".format(val_loss_min))
                    # 保存最优模型。。。
                    print("Saving checkpoint...")
                    torch.save(net, os.path.join(args['save_path'],
                                                 '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(
                                                     args['model_name'],
                                                     i+1, args['local_epoch'],
                                                     args['batch_size'],
                                                     args['learning_rate'],
                                                     args['num_of_clients'],
                                                     args['cfraction'])))

    #  保存运行结果
    np.savetxt("train_acc.csv", train_acc)
    np.savetxt("val_acc.csv", val_acc)
    
    # 训练结束，打印总结信息
    print(f"\n训练完成！")
    print(f"总通信轮次: {args['num_comm']}")
    print(f"最终验证集精度: {100 * val_acc[-1]:.2f}%")
    print(f"日志已保存至: {log_filename}")
    
    # 恢复标准输出和标准错误
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    # 关闭日志文件
    log_file.close()
