import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from collections import Counter
import re


def setup_plotting_environment():
    """设置绘图环境，解决中文显示问题"""
    # 设置中文字体
    plt.rcParams['font.size'] = 12

    # 尝试设置中文字体
    try:
        # Windows 系统中的中文字体
        if os.name == 'nt':  # Windows
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        else:  # Linux/Mac
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']

        plt.rcParams['axes.unicode_minus'] = False

    except Exception as e:
        print(f"字体设置警告: {e}，使用默认英文字体")
        plt.rcParams['font.family'] = 'DejaVu Sans'


setup_plotting_environment()

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 启用cuDNN自动调优
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# 数据预处理类 - 优化版本
class FastTextDataset(Dataset):
    def __init__(self, file_path, seq_length=32, vocab_size=3000):
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 限制数据量以加速训练
        text = text[:500000]  # 只使用前500KB数据

        # 简单的文本清理和分词
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
        tokens = text.lower().split()

        # 构建词汇表
        self.build_vocab(tokens)

        # 将文本转换为索引序列
        self.data = [self.word_to_idx.get(word, self.word_to_idx['<unk>'])
                     for word in tokens if word in self.word_to_idx]

    def build_vocab(self, tokens):
        counter = Counter(tokens)
        common_words = counter.most_common(self.vocab_size - 2)

        self.idx_to_word = ['<pad>', '<unk>'] + [word for word, _ in common_words]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.idx_to_word)}

        print(f"词汇表大小: {len(self.word_to_idx)}")

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y


# 修正的位置编码类
class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 形状变为 [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x的形状: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]  # 修正：使用x.size(1)作为序列长度


# 多头自注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len = q.size(0), q.size(1)

        # 线性变换
        q = self.w_q(q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # 应用注意力权重
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.w_o(output)


# 逐位置前馈网络
class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# 残差连接 + 层归一化
class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feedforward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.attention_sublayer = SublayerConnection(d_model, dropout)
        self.ff_sublayer = SublayerConnection(d_model, dropout)

    def forward(self, x, mask=None):
        x = self.attention_sublayer(x, lambda x: self.attention(x, x, x, mask))
        x = self.ff_sublayer(x, self.feedforward)
        return x


# 完整的Transformer模型（修改以支持消融实验）
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=3,
                 d_ff=512, max_seq_len=50, dropout=0.1,
                 use_positional=True, use_residual=True, use_layernorm=True):
        super(TransformerLM, self).__init__()

        self.d_model = d_model
        self.use_positional = use_positional
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码（可根据消融实验禁用）
        if self.use_positional:
            self.pos_encoding = FixedPositionalEncoding(d_model, max_seq_len)

        self.dropout = nn.Dropout(dropout)

        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 最终层归一化（可根据消融实验禁用）
        if self.use_layernorm:
            self.layer_norm = nn.LayerNorm(d_model)

        self.output_layer = nn.Linear(d_model, vocab_size)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        # 嵌入层
        x = self.embedding(x) * np.sqrt(self.d_model)

        # 位置编码
        if self.use_positional:
            x = self.pos_encoding(x)

        x = self.dropout(x)

        # Transformer层（根据是否使用残差连接调整）
        for layer in self.layers:
            if self.use_residual:
                x = layer(x, mask)
            else:
                # 不使用残差连接
                x = layer.attention_sublayer.norm(x)
                x = layer.attention(x, x, x, mask)
                x = layer.ff_sublayer.norm(x)
                x = layer.feedforward(x)

        # 最终层归一化
        if self.use_layernorm:
            x = self.layer_norm(x)

        return self.output_layer(x)


# 训练函数 - 改为10轮，优化速度
def train_model_10_epochs(model, train_loader, val_loader, vocab_size, device, experiment_name):
    """训练10个epoch的函数，优化速度"""
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
    # 使用更大的学习率加速收敛
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    num_epochs = 10  # 改为10轮

    print(f"开始训练 {experiment_name}，共{num_epochs}个epoch")

    for epoch in range(num_epochs):
        start_time = time.time()
        # 训练阶段
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        batch_count = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)

            # 计算损失
            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # 每50个batch计算一次准确率以节省时间
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    preds = output.argmax(dim=-1)
                    mask = targets != 0
                    total_correct += ((preds == targets) & mask).sum().item()
                    total_tokens += mask.sum().item()

            # 大幅减少打印频率
            if batch_idx % 200 == 0:
                print(
                    f'{experiment_name} - Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        scheduler.step()

        # 计算训练指标
        avg_train_loss = total_loss / batch_count
        train_accuracy = total_correct / max(total_tokens, 1) if total_tokens > 0 else 0

        # 快速验证（只评估前50个batch）
        avg_val_loss, val_accuracy = fast_evaluate(model, val_loader, vocab_size, device, criterion, max_batches=50)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        epoch_time = time.time() - start_time
        print(f'{experiment_name} - Epoch {epoch + 1}/{num_epochs} - 时间: {epoch_time:.1f}秒')
        print(f'训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}')
        print(f'训练准确率: {train_accuracy:.4f}, 验证准确率: {val_accuracy:.4f}')
        print('-' * 50)

    return train_losses, val_losses, train_accs, val_accs


def fast_evaluate(model, data_loader, vocab_size, device, criterion, max_batches=50):
    """快速评估，只评估部分批次"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    batch_count = 0

    with torch.no_grad():
        for data, targets in data_loader:
            if batch_count >= max_batches:
                break

            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            output = model(data)

            loss = criterion(output.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()

            preds = output.argmax(dim=-1)
            mask = targets != 0
            total_correct += ((preds == targets) & mask).sum().item()
            total_tokens += mask.sum().item()

            batch_count += 1

    avg_loss = total_loss / batch_count
    accuracy = total_correct / max(total_tokens, 1) if total_tokens > 0 else 0

    return avg_loss, accuracy


# 消融实验配置
def get_ablation_experiments():
    """完整的消融实验配置"""
    return {
        # 基线模型
        "基线模型": {
            "d_model": 128, "n_heads": 4, "n_layers": 3, "d_ff": 512,
            "use_positional": True, "use_residual": True, "use_layernorm": True
        },

        # 不同头数比较
        "1头注意力": {
            "d_model": 128, "n_heads": 1, "n_layers": 3, "d_ff": 512,
            "use_positional": True, "use_residual": True, "use_layernorm": True
        },
        "2头注意力": {
            "d_model": 128, "n_heads": 2, "n_layers": 3, "d_ff": 512,
            "use_positional": True, "use_residual": True, "use_layernorm": True
        },
        "8头注意力": {
            "d_model": 128, "n_heads": 8, "n_layers": 3, "d_ff": 512,
            "use_positional": True, "use_residual": True, "use_layernorm": True
        },

        # 位置编码消融
        "无位置编码": {
            "d_model": 128, "n_heads": 4, "n_layers": 3, "d_ff": 512,
            "use_positional": False, "use_residual": True, "use_layernorm": True
        },

        # 架构组件消融
        "无残差连接": {
            "d_model": 128, "n_heads": 4, "n_layers": 3, "d_ff": 512,
            "use_positional": True, "use_residual": False, "use_layernorm": True
        },
        "无层归一化": {
            "d_model": 128, "n_heads": 4, "n_layers": 3, "d_ff": 512,
            "use_positional": True, "use_residual": True, "use_layernorm": False
        },

        # 模型大小比较
        "小型模型": {
            "d_model": 64, "n_heads": 2, "n_layers": 2, "d_ff": 256,
            "use_positional": True, "use_residual": True, "use_layernorm": True
        },
        "大型模型": {
            "d_model": 256, "n_heads": 8, "n_layers": 4, "d_ff": 1024,
            "use_positional": True, "use_residual": True, "use_layernorm": True
        },
    }


# 根据消融配置创建模型
def create_model_for_ablation(vocab_size, config_name, ablation_config):
    base_config = {
        "vocab_size": vocab_size,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 3,
        "d_ff": 512,
        "max_seq_len": 50,
        "dropout": 0.1,
        "use_positional": True,
        "use_residual": True,
        "use_layernorm": True
    }

    # 更新配置
    base_config.update(ablation_config)

    return TransformerLM(**base_config), config_name


# 绘制结果函数 - 中文标题
def plot_comprehensive_results(results):
    """绘制全面的结果图表 - 中文标题"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 训练损失
    for config_name, metrics in results.items():
        axes[0, 0].plot(metrics['train_loss'], label=config_name, linewidth=2)
    axes[0, 0].set_title('训练损失')
    axes[0, 0].set_xlabel('训练轮次')
    axes[0, 0].set_ylabel('损失值')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    # 验证损失
    for config_name, metrics in results.items():
        axes[0, 1].plot(metrics['val_loss'], label=config_name, linewidth=2)
    axes[0, 1].set_title('验证损失')
    axes[0, 1].set_xlabel('训练轮次')
    axes[0, 1].set_ylabel('损失值')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    # 训练准确率
    for config_name, metrics in results.items():
        axes[1, 0].plot(metrics['train_acc'], label=config_name, linewidth=2)
    axes[1, 0].set_title('训练准确率')
    axes[1, 0].set_xlabel('训练轮次')
    axes[1, 0].set_ylabel('准确率')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    # 验证准确率
    for config_name, metrics in results.items():
        axes[1, 1].plot(metrics['val_acc'], label=config_name, linewidth=2)
    axes[1, 1].set_title('验证准确率')
    axes[1, 1].set_xlabel('训练轮次')
    axes[1, 1].set_ylabel('准确率')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()

    plt.show()


# 绘制特定比较图表 - 中文标题
def plot_specific_comparisons(results, comparison_groups):
    """绘制特定组的比较图表 - 中文标题"""
    for group_name, configs in comparison_groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 验证损失比较
        for config_name in configs:
            if config_name in results:
                axes[0].plot(results[config_name]['val_loss'], label=config_name, linewidth=2)
        axes[0].set_title(f'{group_name} - 验证损失')
        axes[0].set_xlabel('训练轮次')
        axes[0].set_ylabel('损失值')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 验证准确率比较
        for config_name in configs:
            if config_name in results:
                axes[1].plot(results[config_name]['val_acc'], label=config_name, linewidth=2)
        axes[1].set_title(f'{group_name} - 验证准确率')
        axes[1].set_xlabel('训练轮次')
        axes[1].set_ylabel('准确率')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据路径 - 请根据实际情况修改
    train_file = "data/wikitext-2-raw/wiki.train.tokens"  # 请修改为实际路径
    val_file = "data/wikitext-2-raw/wiki.valid.tokens"  # 请修改为实际路径

    # 创建数据集（使用较小配置以加速实验）
    print("加载数据集...")
    train_dataset = FastTextDataset(train_file, seq_length=32, vocab_size=3000)
    val_dataset = FastTextDataset(val_file, seq_length=32, vocab_size=3000)

    # 根据GPU内存设置batch_size
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb >= 8:
            batch_size = 128
        elif gpu_memory_gb >= 6:
            batch_size = 64
        else:
            batch_size = 32
    else:
        batch_size = 16

    print(f"使用batch_size: {batch_size}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # 增加数据加载线程
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    vocab_size = len(train_dataset.word_to_idx)
    print(f"词汇表大小: {vocab_size}")

    # 消融实验配置
    ablation_configs = get_ablation_experiments()

    # 选择关键实验以节省时间
    key_experiments = {
        "基线模型": ablation_configs["基线模型"],
        "1头注意力": ablation_configs["1头注意力"],
        "2头注意力": ablation_configs["2头注意力"],
        "8头注意力": ablation_configs["8头注意力"],
        "无位置编码": ablation_configs["无位置编码"],
        "无残差连接": ablation_configs["无残差连接"],
        "小型模型": ablation_configs["小型模型"],
    }

    results = {}

    # 运行关键实验
    for config_name, ablation_config in key_experiments.items():
        print(f"\n{'=' * 60}")
        print(f"开始实验: {config_name}")
        print(f"配置: {ablation_config}")
        print(f"{'=' * 60}")

        # 创建模型
        model, exp_name = create_model_for_ablation(vocab_size, config_name, ablation_config)
        model = model.to(device)

        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

        # 训练模型（10个epoch）
        train_loss, val_loss, train_acc, val_acc = train_model_10_epochs(
            model, train_loader, val_loader, vocab_size, device, config_name
        )

        # 保存结果
        results[config_name] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }

        # 保存模型
        torch.save(model.state_dict(), f"transformer_{config_name}.pth")

    # 绘制综合结果
    plot_comprehensive_results(results)
    plt.savefig("comprehensive_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    # 绘制特定比较组
    comparison_groups = {
        "不同头数比较": ["1头注意力", "2头注意力", "8头注意力", "基线模型"],
        "架构组件比较": ["基线模型", "无位置编码", "无残差连接", "小型模型"]
    }
    plot_specific_comparisons(results, comparison_groups)

    # 打印最终结果比较
    print("\n最终结果比较:")
    print("-" * 100)
    print(f"{'配置':<15} {'训练损失':<10} {'验证损失':<10} {'训练准确率':<12} {'验证准确率':<12} {'轮次数':<8}")
    print("-" * 100)

    for config_name, metrics in results.items():
        train_loss_final = metrics['train_loss'][-1] if len(metrics['train_loss']) > 0 else float('inf')
        val_loss_final = metrics['val_loss'][-1] if len(metrics['val_loss']) > 0 else float('inf')
        train_acc_final = metrics['train_acc'][-1] if len(metrics['train_acc']) > 0 else 0
        val_acc_final = metrics['val_acc'][-1] if len(metrics['val_acc']) > 0 else 0
        epochs_trained = len(metrics['val_loss'])

        print(f"{config_name:<15} {train_loss_final:<10.4f} {val_loss_final:<10.4f} "
              f"{train_acc_final:<12.4f} {val_acc_final:<12.4f} {epochs_trained:<8}")

    # 分析位置编码的影响
    if '基线模型' in results and '无位置编码' in results:
        baseline_acc = results['基线模型']['val_acc'][-1]
        no_pos_acc = results['无位置编码']['val_acc'][-1]
        pos_impact = (baseline_acc - no_pos_acc) / no_pos_acc * 100 if no_pos_acc > 0 else 0

        print(f"\n位置编码影响分析:")
        print(f"基线模型准确率: {baseline_acc:.4f}")
        print(f"无位置编码准确率: {no_pos_acc:.4f}")
        print(f"位置编码带来的提升: {pos_impact:+.2f}%")

        if pos_impact > 0:
            print("位置编码对模型性能有积极影响")
        else:
            print("位置编码在此配置下未显示明显优势")

    # 分析不同头数的影响
    head_configs = [
        ("1头注意力", 1),
        ("2头注意力", 2),
        ("基线模型", 4),
        ("8头注意力", 8)
    ]
    head_results = []

    for config_name, n_heads in head_configs:
        if config_name in results:
            accuracy = results[config_name]['val_acc'][-1]
            head_results.append((n_heads, accuracy, config_name))

    if len(head_results) > 1:
        print(f"\n不同头数影响分析:")
        for n_heads, accuracy, config_name in sorted(head_results):
            print(f"{config_name} ({n_heads}头): {accuracy:.4f}")

    # 分析模型大小的影响
    if '基线模型' in results and '小型模型' in results and '大型模型' in results:
        baseline_acc = results['基线模型']['val_acc'][-1]
        small_acc = results['小型模型']['val_acc'][-1]
        large_acc = results['大型模型']['val_acc'][-1] if '大型模型' in results else 0

        print(f"\n模型大小影响分析:")
        print(f"小型模型准确率: {small_acc:.4f}")
        print(f"基线模型准确率: {baseline_acc:.4f}")
        if '大型模型' in results:
            print(f"大型模型准确率: {large_acc:.4f}")


if __name__ == "__main__":
    main()