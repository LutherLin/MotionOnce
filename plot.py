import torch
import matplotlib.pyplot as plt
import numpy as np

# 加载.pt文件
tensor = torch.load('attn.pt')

# 确保数据在CPU上，避免GPU相关的错误
if tensor.is_cuda:
    tensor = tensor.cpu()
triu_indices = torch.triu_indices(21, 21, offset=1)
for i in range(tensor.size(1)):
    # 使用索引将上三角部分置为 0
    tensor[0, i, triu_indices[0], triu_indices[1]] = 0
# 检查张量的形状是否正确
assert tensor.shape == (1, 6, 21, 21), "The shape of the tensor is not correct!"

# 创建一个画布，用于放置6个子图
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()  # 将axs展平，以便更容易地迭代

# 遍历第二维度的每一个切片，并绘制热图
for i in range(6):
    # 选择当前的切片
    slice_i = tensor[0, i, :, :].numpy()
    
    # 绘制热图
    im = axs[i].imshow(slice_i, cmap='hot', interpolation='nearest')
    
    # 可选：添加颜色条
    cbar = fig.colorbar(im, ax=axs[i])
    cbar.ax.set_ylabel('Attention Weight', rotation=90)
    
    # 设置标题
    axs[i].set_title(f'Attention Map {i+1}')

# 调整子图间距
plt.tight_layout()
plt.savefig('attention_maps.png', dpi=300, bbox_inches='tight')
# 显示图像
plt.show()