import numpy as np
import gzip
import os
from sklearn.utils import shuffle
from torchvision import datasets, transforms
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'mnist':
            self.mnistDataSetConstruct(isIID)
        elif self.name == 'cifar10':
            self.cifarDataSetConstruct(isIID)
        elif self.name == 'mitbih':
            self.mitbihDataSetConstruct(isIID)
        else:
            raise ValueError(f"Unsupported dataset: {self.name}")

    def mitbihDataSetConstruct(self, isIID):
        """
        构建MIT-BIH心律不齐数据集
        数据集包含5种心律类型：
        - N: 正常心律
        - S: 室上性异位心律  
        - V: 室性异位心律
        - F: 心房颤动
        - Q: 未知心律
        """
        data_dir = r'./data/MIT-BIH'
        
        # 数据文件路径
        train_data_path = os.path.join(data_dir, 'mitbih_train.csv')
        test_data_path = os.path.join(data_dir, 'mitbih_test.csv')
        
        # 检查数据文件是否存在
        if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
            raise FileNotFoundError(
                f"MIT-BIH数据文件不存在！\n"
                f"请确保以下文件存在：\n"
                f"- {train_data_path}\n"
                f"- {test_data_path}"
            )
        
        print("正在加载MIT-BIH数据集...")
        
        # 加载CSV文件
        train_df = pd.read_csv(train_data_path, header=None)
        test_df = pd.read_csv(test_data_path, header=None)
        
        # 分离特征和标签
        train_data = train_df.iloc[:, :-1].values.astype(np.float32)
        train_labels = train_df.iloc[:, -1].values.astype(np.int64)
        test_data = test_df.iloc[:, :-1].values.astype(np.float32)
        test_labels = test_df.iloc[:, -1].values.astype(np.int64)
        
        # 数据标准化
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
        
        # 转换标签为one-hot编码
        train_labels_onehot = dense_to_one_hot(train_labels, num_classes=5)
        test_labels_onehot = dense_to_one_hot(test_labels, num_classes=5)
        
        # 验证数据
        assert train_data.shape[0] == train_labels_onehot.shape[0]
        assert test_data.shape[0] == test_labels_onehot.shape[0]
        
        self.train_data_size = train_data.shape[0]
        self.test_data_size = test_data.shape[0]
        
        # 根据是否IID分布数据
        if isIID:
            # 打乱顺序
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_data[order]
            self.train_label = train_labels_onehot[order]
        else:
            # 按标签排序（Non-IID）
            order = np.argsort(train_labels)
            self.train_data = train_data[order]
            self.train_label = train_labels_onehot[order]
        
        self.test_data = test_data
        self.test_label = test_labels_onehot
        
        print(f"MIT-BIH数据集加载完成:")
        print(f"训练集大小: {self.train_data_size}")
        print(f"测试集大小: {self.test_data_size}")
        print(f"特征维度: {train_data.shape[1]}")
        print(f"类别数: 5")
   
    
    
    def mnistDataSetConstruct(self, isIID):
        data_dir = r'./data/MNIST'
        # 选定图片路径
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        # 从.gz中提取图片
        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        # mnist黑白图片通道为1
        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        # 图片展平
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])
        
        # 标准化处理
        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        # 是否独立同分布
        if isIID:
            # 打乱顺序
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            # 按照0——9顺序排列
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels

    def cifarDataSetConstruct(self, isIID):
        data_dir = r'./data/'
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
			transforms.RandomCrop(32, padding=4),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

        transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
										transform=transform_train)
        eval_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False, num_workers=2)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1000, shuffle=False, num_workers=2)
        
        # 处理训练数据
        train_data = []
        train_labels_list = []
        for images, labels in train_loader:
            train_data.append(images.numpy())
            train_labels_list.append(labels.numpy())
        train_images = np.concatenate(train_data)
        train_labels = dense_to_one_hot(np.concatenate(train_labels_list))

        # 处理测试数据
        test_data = []
        test_labels_list = []
        for images, labels in eval_loader:
            test_data.append(images.numpy())
            test_labels_list.append(labels.numpy())
        test_images  = np.concatenate(test_data)
        test_labels = dense_to_one_hot(np.concatenate(test_labels_list))

        # 验证数据导入无误
        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        # cifar彩色图片通道为3
        assert train_images.shape[1] == 3
        assert test_images.shape[1] == 3

        # 是否独立同分布
        if isIID:
            # 打乱顺序
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            # 按照0——9顺序排列
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels

# 比特流读取
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

# 提取图片
def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data

# 标签one-hot编码
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

# 提取标签
def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)

# 测试代码
if __name__ == "__main__":
    # 测试MIT-BIH数据集
    print("测试MIT-BIH数据集...")
    mitbih_dataset = GetDataSet('mitbih', True)  # IID分布
    
    print(f"训练数据形状: {mitbih_dataset.train_data.shape}")
    print(f"训练标签形状: {mitbih_dataset.train_label.shape}")
    print(f"测试数据形状: {mitbih_dataset.test_data.shape}")
    print(f"测试标签形状: {mitbih_dataset.test_label.shape}")
    
    # 测试Non-IID分布
    print("\n测试Non-IID分布...")
    mitbih_dataset_noniid = GetDataSet('mitbih', False)
    
    # 检查数据类型
    if (type(mitbih_dataset.train_data) is np.ndarray and 
        type(mitbih_dataset.test_data) is np.ndarray and
        type(mitbih_dataset.train_label) is np.ndarray and 
        type(mitbih_dataset.test_label) is np.ndarray):
        print("数据类型正确: numpy ndarray")
    else:
        print("数据类型错误")
