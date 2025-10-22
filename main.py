import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np

from data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from model import CNNModel
from test import test

# ======================
# 1. 参数配置
# ======================

source_dataset_name = 'MNIST'
target_dataset_name = 'mnist_m'
source_image_root = os.path.join('dataset', source_dataset_name)
target_image_root = os.path.join('dataset', target_dataset_name)
model_root = 'models'

cuda = True
cudnn.benchmark = True

lr = 1e-3
batch_size = 128
image_size = 28
n_epoch = 100

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# ======================
# 2. 数据预处理
# ======================

img_transform_source = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

img_transform_target = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# ======================
# 3. 数据加载（⚠️ 注意：num_workers 在 Windows 上建议先设为 0）
# ======================

dataset_source = datasets.MNIST(
    root='dataset',
    train=True,
    transform=img_transform_source,
    download=True
)

# ⚠️ Windows 用户：num_workers 设为 0 可避免多进程启动错误，稳定后再尝试 2 或 4
dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0  # ✅ 改为 0 可解决 Windows 报错
)

train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')

dataset_target = GetLoader(
    data_root=os.path.join(target_image_root, 'mnist_m_train'),
    data_list=train_list,
    transform=img_transform_target
)

# ⚠️ 同上，Windows 下建议 num_workers=0
dataloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0  # ✅ 改为 0 可解决 Windows 报错
)

# ======================
# 4. 模型、优化器、损失函数
# ======================

my_net = CNNModel()

optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# ======================
# 5. 训练函数（封装在 main() 中）
# ======================

def main():
    best_accu_t = 0.0
    for epoch in range(n_epoch):
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        for i in range(len_dataloader):
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # ---- Source Data ----
            data_source = next(data_source_iter)
            s_img, s_label = data_source

            my_net.zero_grad()
            batch_size = len(s_label)

            domain_label = torch.zeros(batch_size).long()

            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                domain_label = domain_label.cuda()

            class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # ---- Target Data ----
            data_target = next(data_target_iter)
            t_img, _ = data_target

            batch_size = len(t_img)
            domain_label = torch.ones(batch_size).long()

            if cuda:
                t_img = t_img.cuda()
                domain_label = domain_label.cuda()

            _, domain_output = my_net(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)

            # ---- 总 Loss ----
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()

            # ---- 打印日志（可选优化：每N个iter打印一次）----
            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %.4f, err_s_domain: %.4f, err_t_domain: %.4f' %
                            (epoch, i + 1, len_dataloader,
                             err_s_label.item(), err_s_domain.item(), err_t_domain.item()))
            sys.stdout.flush()

            # ---- 每个 iter 都保存模型（可选：可改为每 epoch 保存一次以减少文件数量）----
            torch.save(my_net, os.path.join(model_root, 'mnist_mnistm_model_epoch_current.pth'))

        # ---- 每个 Epoch 测试 & 保存最佳模型 ----
        print('\n')
        accu_s = test(source_dataset_name)
        print('Accuracy of the %s dataset: %f' % ('mnist', accu_s))
        accu_t = test(target_dataset_name)
        print('Accuracy of the %s dataset: %f\n' % ('mnist_m', accu_t))

        if accu_t > best_accu_t:
            best_accu_s = accu_s
            best_accu_t = accu_t
            torch.save(my_net, os.path.join(model_root, 'mnist_mnistm_model_epoch_best.pth'))

    # ---- 训练结束总结 ----
    print('============ Summary =============')
    print('Accuracy of the %s dataset: %f' % ('mnist', best_accu_s))
    print('Accuracy of the %s dataset: %f' % ('mnist_m', best_accu_t))
    print('Best model saved in: ' + os.path.join(model_root, 'mnist_mnistm_model_epoch_best.pth'))

# ======================
# 6. 程序入口（Windows 必须！）
# ======================
if __name__ == '__main__':
    main()