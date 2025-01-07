import torch
import torch.nn as nn

def compute_attributions(model, reference_input, actual_inputs):
    # 前向传播得到输出
    all_attri = []
    for i in range(actual_inputs.size(0)):
        actual_input = actual_inputs[i:i+1, :]
        def forward_pass(model, x):
            inputs = []
            outputs = []
            for layer in model.children():
                inputs.append(x)
                x = layer(x)
                outputs.append(x)
            return inputs, outputs

        # Perform forward passes for both reference and actual inputs
        ref_inputs, ref_outputs = forward_pass(model, reference_input)
        act_inputs, act_outputs = forward_pass(model, actual_input)
        # 计算输入差异
        delta_input = actual_input - reference_input
        # 初始化归因
        attributions = delta_input.clone()
        # 逐层计算归因
        layer_idx = 0
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                weight = layer.weight
                # 线性层的权重
                if layer_idx == 0:
                    # 更新归因
                    attributions = torch.matmul(weight, torch.diag_embed(attributions.float()))
                else:
                    attributions = torch.matmul(weight, attributions)
            elif isinstance(layer, nn.ReLU):
                # 计算非线性层的输入和输出差异
                delta_input_relu = act_inputs[layer_idx] - ref_inputs[layer_idx]
                delta_output_relu = act_outputs[layer_idx] - ref_outputs[layer_idx]

                # 计算非线性层的乘子
                # relu_multiplier = torch.where(
                #     delta_input_relu != 0,
                #     delta_output_relu / delta_input_relu,
                #     torch.tensor(1.0).to('cuda:0')  # 当输入差异为 0 时，乘子为 1
                # )
                # relu_multiplier = torch.where(
                #     torch.abs(delta_input_relu) < 0.001,  # 条件：绝对值小于 0.001
                #     torch.tensor(1.0).to('cuda:0'),  # 当条件为真时，乘子为 1
                #     delta_output_relu / delta_input_relu  # 否则，使用 delta_output_relu / delta_input_relu
                # )
                relu_multiplier = torch.where(
                    torch.abs(delta_input_relu) < 0.001,  # 条件：绝对值小于 0.001
                    torch.tensor(1.0).to('cuda:0'),  # 当条件为真时，乘子为 1
                    delta_output_relu / (delta_input_relu + 1e-8)  # 防止除以零，增加一个小常数
                )

                # 更新归因
                attributions = torch.matmul(torch.diag_embed(relu_multiplier.float()), attributions.float())
            layer_idx = layer_idx + 1
        all_attri.append(torch.mean(attributions, dim=0))
    return all_attri

def gradient_method(model, baselines, input_tensor):
    """
    计算梯度显著图（Gradient Saliency），支持批量输入，并保留梯度信息以进行后续优化。

    参数:
        model (nn.Module): PyTorch模型。
        input_tensor (torch.Tensor): 输入张量，形状为 (batch_size, n_features)。

    返回:
        feature_importance (torch.Tensor): 特征重要性，形状为 (batch_size, output_dim, n_features)。
        gradients (torch.Tensor): 原始梯度值，形状为 (batch_size, output_dim, n_features)。
    """
    # 确保输入张量需要梯度
    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    # 前向传播
    output = model(input_tensor)  # 输出形状: (batch_size, output_dim)

    batch_size, output_dim = output.shape
    n_features = input_tensor.shape[1]

    # 初始化梯度张量
    gradients = torch.zeros(batch_size, output_dim, n_features, device=input_tensor.device)

    # 使用 torch.autograd.grad 计算梯度，保持计算图连贯
    for i in range(output_dim):
        # 对每个输出类别，计算梯度
        grad_output = output[:, i].sum()  # 将当前类别的所有样本输出求和，得到一个标量

        # 计算梯度
        grad = torch.autograd.grad(
            outputs=grad_output,
            inputs=input_tensor,
            retain_graph=True,
            create_graph=True
        )[0]  # grad 的形状: (batch_size, n_features)

        gradients[:, i, :] = grad
    return gradients * (input_tensor-baselines).unsqueeze(1)

