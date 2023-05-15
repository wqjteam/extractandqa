import torch
from torch.optim.lr_scheduler import LambdaLR

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# 自定义学习率调整函数
def lr_lambda(epoch):
    if epoch < 10:
        return 1
    elif epoch < 20:
        return 0.5
    else:
        return 0.1

# 学习率调整器
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# 训练循环
for epoch in range(num_epochs):
    train_loss = 0
    for input_data, task1_target, task2_target in train_loader:
        optimizer.zero_grad()
        task1_output, task2_output = model(input_data)
        task1_loss = loss_function1(task1_output, task1_target)
        task2_loss = loss_function2(task2_output, task2_target)
        total_loss = task1_loss + task2_loss
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()

    # 调整学习率
    scheduler.step()

    print('Epoch: {}, Train Loss: {:.4f}'.format(epoch, train_loss))
