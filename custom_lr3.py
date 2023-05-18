# 在使用torch的BERT进行多任务学习时，可以使用get_cosine_schedule_with_warmup和LambdaLR等方法来进行预训练，并结合AdaptiveLoss来对单个loss进行自适应减小。
#
# 具体实现步骤如下：
#
# 定义损失函数和优化器
# 对于多任务学习，需要定义多个损失函数，可以使用nn.ModuleList来封装多个损失函数，例如：
from torch import nn


class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks):
        super(MultiTaskLoss, self).__init__()
        self.losses = nn.ModuleList([nn.CrossEntropyLoss() for _ in range(num_tasks)])

    def forward(self, outputs, targets):
        total_loss = 0
        for i, output in enumerate(outputs):
            total_loss += self.losses[i](output, targets[i])
        return total_loss


# 然后定义优化器，例如使用Adam优化器：
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


# 预训练
# 在进行预训练时，可以使用get_cosine_schedule_with_warmup和LambdaLR等方法来调整学习率和权重衰减。例如：

total_steps = num_epochs * len(train_loader)
warmup_steps = int(warmup_ratio * total_steps)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
lambda_lr = lambda epoch: 1 / (1 + decay_rate * epoch)
scheduler_lambda = LambdaLR(optimizer, lr_lambda=lambda_lr)

for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = MultiTaskLoss(num_tasks)(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        scheduler_lambda.step()



# 在训练过程中，每个epoch结束后，可以计算每个任务的单个loss，
# 并根据需要进行自适应减小。例如，可以计算每个任务的平均loss，然后根据某个阈值进行判断是否需要减小对应任务的权重，具体代码实现如下：
def adaptive_loss(losses, prev_losses, prev_weights, threshold):
    if not prev_losses:
        return [1.0] * len(losses)

    weights = []
    for i, loss in enumerate(losses):
        if loss / prev_losses[i] < threshold:
            weights.append(prev_weights[i] / 2)
        else:
            weights.append(prev_weights[i])
    return weights


prev_losses, prev_weights = None, None
for epoch in range(num_epochs):
    losses = [0.0] * num_tasks
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = MultiTaskLoss(num_tasks)(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        scheduler_lambda.step()

        for i in range(num_tasks):
            losses[i] += nn.CrossEntropyLoss()(outputs[i], targets[i]).item()

    if prev_losses and prev_weights:
        weights = adaptive_loss(losses, prev_losses, prev_weights, threshold)
        for i, w in enumerate(weights):
            losses[i] *= w



