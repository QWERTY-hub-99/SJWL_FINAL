import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


if __name__ == "__main__":
    # 定义归一化函数
    ###################################【配置GPU环境】###################################
    # 设置设备为GPU（如果可用）
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print()
    print("##########################GPU信息###########################")
    print(f"可用GPU数量：{torch.cuda.device_count()}")
    print("############################################################")
    print()


    ###################################【加载CIFAR-100数据集】###################################
    # 定义归一化函数
    transform = transforms.Compose([
        transforms.ToTensor(), # 转化到[0, 1]的区间
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 图像归一化，转化到[-1, 1]的区间
    ])

    # 加载CIFAR-100数据集
    batch_size = 64 ##########################################
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print()
    print("#########################数据集信息#########################")
    # 打印训练集样本信息
    print("训练集样本数量:", len(train_dataset))
    print("训练集类别数量:", len(train_dataset.classes))
    #print("训练集类别列表:", train_dataset.classes)

    # 打印测试集样本信息
    print("测试集样本数量:", len(test_dataset))
    print("测试集类别数量:", len(test_dataset.classes))
    #print("测试集类别列表:", test_dataset.classes)

    # 展示数据集前2个样本的信息
    for i in range(2):
        image, label = train_dataset[i]
        print()
        print(f"样本{i+1}")
        print(f"图像大小：{image.size()}")
        print(f"类别：{label}")
    print("############################################################")
    print()


    ##################################【Transformer模型】###################################
    class Transformer_22M_Model(nn.Module):
        def __init__(self, num_classes):
            super(Transformer_22M_Model, self).__init__()
            self.patch_embedding = nn.Linear(3 * 32 * 32, 512)  # Patch Embedding的通道数
            self.dropout = nn.Dropout(0.5)  # 添加一个dropout层
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=512, nhead=16, dim_feedforward=1024),  # Multi-head Attention的头数和注意力尺寸
                num_layers=32  # Transformer Encoder层数
            )
            self.fc = nn.Linear(512, num_classes)  # num_classes为分类任务的类别数

        def forward(self, x):
            x = x.view(x.size(0), -1)  # 展平图像数据为向量形式
            x = self.patch_embedding(x)
            x = self.dropout(x)
            x = x.view(x.size(0), 8, 8, -1)  # 调整维度为 (batch_size, 8, 8, channels)
            x = x.permute(0, 3, 1, 2)  # 将维度调整为 (batch_size, channels, 8, 8)
            x = x.reshape(x.size(0), -1)  # 将维度调整为 (batch_size, channels * 8 * 8)
            x = self.transformer_encoder(x)
            # x = x.mean(dim=0)  # 对序列维度取平均
            x = self.fc(x)
            return x

        def count_parameters(self):
            # 计算模型参数数量
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # 创建模型实例
    transformer_22M_model = Transformer_22M_Model(num_classes=100)

    print()
    print("##########################模型信息##########################")
    # 统计参数数量
    params_num = transformer_22M_model.count_parameters()
    print(f"模型的参数数量：{params_num}")
    print("############################################################")
    print()


    # 创建模型实例
    transformer_22M_model = nn.DataParallel(Transformer_22M_Model(num_classes=100), device_ids=[0, 1, 2, 3])
    transformer_22M_model = transformer_22M_model.to(device)

    checkpoint = False

    # 加载模型checkpoint
    if checkpoint == True:
        checkpoint = torch.load('checkpoint.pth')
        transformer_22M_model.load_state_dict(checkpoint['model_state_dict'])
        epoch_start = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        train_accuracies = checkpoint['train_accuracies']
        test_loss = checkpoint['test_loss']
        test_accuracy = checkpoint['test_accuracy']
        precise = checkpoint['precise']
        recall = checkpoint['recall']
        f1 = checkpoint['f1']
    else:
        epoch_start = 1
        train_losses = []
        train_accuracies = []
        test_loss = []
        test_accuracy = []
        precise = []
        recall = []
        f1 = []

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(transformer_22M_model.parameters(), lr=0.1)

    epoch_num = 1000

    # 训练循环
    for epoch in tqdm(range(epoch_start, epoch_start + epoch_num)):
        # 将模型设置为训练模式
        transformer_22M_model.train()

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = transformer_22M_model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # 保存训练损失函数和准确率
            train_losses.append(loss.item())
            _, predicted = outputs.max(1)
            total = targets.size(0)
            correct = predicted.eq(targets).sum().item()
            train_accuracies.append(correct / total)

        # 测试
        transformer_22M_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        predicted_labels = []
        true_labels = []

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                data = data.to(device)
                targets = targets.to(device)

                outputs = transformer_22M_model(data)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                predicted_labels.extend(predicted.tolist())
                true_labels.extend(targets.tolist())

        test_loss /= len(test_loader)
        test_accuracy = correct / total

        # 转换为NumPy数组
        true_labels = torch.tensor(true_labels).numpy()
        predicted_labels = torch.tensor(predicted_labels).numpy()

        # 计算精确率、召回率和F1分数
        precise = precision_score(true_labels, predicted_labels, average='macro')
        recall = recall_score(true_labels, predicted_labels, average='macro')
        f1 = f1_score(true_labels, predicted_labels, average='macro')

        print('Epoch: {}'.format(epoch+1))
        print('Train Loss: {:.6f}'.format(train_losses[-1]))
        print('Train Accuracy: {:.6f}'.format(train_accuracies[-1]))
        print('Test Loss: {:.6f}'.format(test_loss))
        print('Test Accuracy: {:.6f}'.format(test_accuracy))
        print('Precise: {:.6f}'.format(precise))
        print('Recall: {:.6f}'.format(recall))
        print('F1 Score: {:.6f}'.format(f1))

        if epoch % 500 == 0:
            save_path = 'model_output/model_epoch{}.pth'.format(epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer_22M_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'precise': precise,
                'recall': recall,
                'f1': f1
            }, save_path)