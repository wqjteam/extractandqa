import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, get_cosine_schedule_with_warmup

# 定义多任务损失函数
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
criterion3 = nn.BCEWithLogitsLoss()

# 定义优化器
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# 定义学习率调整策略
num_training_steps = len(train_loader) * num_epochs
scheduler1 = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
scheduler2 = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + epoch))

for epoch in range(num_epochs):
    # 训练模型
    for batch in train_loader:
        inputs, task1_labels, task2_labels, task3_labels = batch
        outputs = model(inputs)
        loss1 = criterion1(outputs[0], task1_labels)
        loss2 = criterion2(outputs[1], task2_labels)
        loss3 = criterion3(outputs[2], task3_labels)
        # 对第一个任务的损失函数进行自适应调整
        weight_factor = max(0, 1 - epoch/num_epochs)
        loss = loss1 * weight_factor + loss2 + loss3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 调整学习率
        scheduler1.step()
        scheduler2.step()
