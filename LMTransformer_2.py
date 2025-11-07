import os
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm


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


class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # 确保mask维度与scores匹配
            if mask.dim() == 2:
                # 如果是2D mask (seq_len, seq_len)，扩展维度以匹配多头注意力
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            elif mask.dim() == 3:
                # 如果是3D mask (batch_size, seq_len, seq_len)，增加头维度
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)

            # 应用掩码（将mask为0的位置设为负无穷）
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.softmax(scores)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = SelfAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.size()

        # 线性变换并重塑为多头
        Q = self.W_q(q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 如果提供了掩码，需要扩展以匹配多头注意力的形状
        if mask is not None:
            # 确保mask有正确的维度 (batch_size, n_heads, seq_len, seq_len)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            # 现在扩展到多头
            mask = mask.expand(batch_size, self.n_heads, seq_len, seq_len)

        out, attn = self.attention(Q, K, V, mask)

        # 重塑回原始形状
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_k)
        out = self.fc(out)
        out = self.dropout(out)
        return self.norm(out + q), attn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        out = torch.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return self.norm(out + residual)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class EncoderBlock(nn.Module):
    """标准的Transformer编码器块"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, src_mask=None):
        # 自注意力层
        out, attn_weights = self.self_attn(x, x, x, mask=src_mask)
        # 前馈网络
        out = self.ffn(out)
        return out, attn_weights


class EncoderOnlyTransformer(nn.Module):
    """仅包含编码器的Transformer，用于分类或序列标注任务"""

    def __init__(self, vocab_size, num_classes=2, d_model=512, n_heads=8,
                 num_layers=6, d_ff=2048, dropout=0.1, max_len=5000,
                 task_type='classification'):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.task_type = task_type

        # 词嵌入和位置编码
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # 分类头
        if task_type == 'classification':
            # 用于序列分类（如情感分析）
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes)
            )
        elif task_type == 'sequence_labeling':
            # 用于序列标注（如命名实体识别）
            self.classifier = nn.Linear(d_model, num_classes)

        # 初始化权重
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        # 词嵌入和位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 编码器层
        attention_weights = []
        for layer in self.encoder_layers:
            x, attn = layer(x, mask)
            attention_weights.append(attn)

        # 根据任务类型输出
        if self.task_type == 'classification':
            # 使用[CLS] token或平均池化进行分类
            # 这里使用平均池化
            x = x.mean(dim=1)  # (batch_size, d_model)
            output = self.classifier(x)
        elif self.task_type == 'sequence_labeling':
            # 序列标注，每个token都有输出
            output = self.classifier(x)  # (batch_size, seq_len, num_classes)

        return {
            'output': output,
            'attention_weights': attention_weights
        }


class WikiText2Dataset(Dataset):
    """WikiText-2数据集加载器，用于语言建模任务"""

    def __init__(self, dataset_type='train', seq_length=128, vocab_size=10000, data_dir="data/wikitext-2-raw"):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.data_dir = data_dir

        print(f"加载WikiText-2 {dataset_type}数据集（本地文件）...")

        # 文件映射
        file_map = {
            'train': 'wiki.train.tokens',
            'validation': 'wiki.valid.tokens',
            'test': 'wiki.test.tokens'
        }

        file_path = os.path.join(data_dir, file_map[dataset_type])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")

        # 读取原始文本
        self.lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith(' = '):
                    self.lines.append(stripped_line)

        print(f"成功加载 {len(self.lines)} 行数据")
        self._build_vocab()
        self._preprocess_data()
        print(f"数据集处理完成，共有 {len(self.token_ids)} 个token")
        print(f"词表大小: {len(self.vocab)}")

    def _build_vocab(self):
        """构建词表"""
        all_tokens = []
        for line in self.lines:
            tokens = line.split()
            all_tokens.extend(tokens)

        word_counts = Counter(all_tokens)
        most_common = word_counts.most_common(self.vocab_size - 4)  # 保留空间给特殊token

        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.vocab = special_tokens + [word for word, _ in most_common]
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}

    def _preprocess_data(self):
        """预处理文本数据为token ID序列"""
        self.token_ids = []
        for line in self.lines:
            tokens = line.split()
            token_ids = [self.word2idx['<sos>']]  # 开始标记
            for token in tokens:
                if token in self.word2idx:
                    token_ids.append(self.word2idx[token])
                else:
                    token_ids.append(self.word2idx['<unk>'])  # 未知词
            token_ids.append(self.word2idx['<eos>'])  # 结束标记
            self.token_ids.extend(token_ids)

    def __len__(self):
        return len(self.token_ids) // self.seq_length

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1  # +1 用于目标序列
        sequence = self.token_ids[start_idx:end_idx]

        # 填充序列
        if len(sequence) < self.seq_length + 1:
            padding_needed = self.seq_length + 1 - len(sequence)
            sequence = sequence + [self.word2idx['<pad>']] * padding_needed

        # 输入序列和目标序列（目标序列是输入序列向右移动一位）
        input_seq = torch.tensor(sequence[:self.seq_length], dtype=torch.long)
        target_seq = torch.tensor(sequence[1:self.seq_length + 1], dtype=torch.long)

        return input_seq, target_seq


def create_data_loaders(batch_size=32, seq_length=128, vocab_size=10000, data_dir="data/wikitext-2-raw"):
    """创建训练、验证和测试数据加载器"""
    train_dataset = WikiText2Dataset('train', seq_length, vocab_size, data_dir)
    val_dataset = WikiText2Dataset('validation', seq_length, vocab_size, data_dir)
    test_dataset = WikiText2Dataset('test', seq_length, vocab_size, data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.vocab


def create_padding_mask(seq, pad_token=0):
    """创建填充掩码（pad位置为0，其他位置为1）"""
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    return mask


class TransformerTrainer:
    def __init__(self, model, train_loader, val_loader, vocab_size, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_size = vocab_size
        self.device = device

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding token
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.8)

        self.train_losses = []
        self.val_losses = []
        self.perplexities = []

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch_idx, (input_seq, target_seq) in enumerate(progress_bar):
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)

            # 创建填充掩码
            mask = create_padding_mask(input_seq).to(self.device)

            self.optimizer.zero_grad()
            output_dict = self.model(input_seq, mask)
            output = output_dict['output']

            # 计算损失
            loss = self.criterion(output.view(-1, self.vocab_size), target_seq.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for input_seq, target_seq in self.val_loader:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)

                # 创建填充掩码
                mask = create_padding_mask(input_seq).to(self.device)

                output_dict = self.model(input_seq, mask)
                output = output_dict['output']
                loss = self.criterion(output.view(-1, self.vocab_size), target_seq.view(-1))
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        perplexity = math.exp(avg_loss)  # 困惑度 = exp(损失)

        self.val_losses.append(avg_loss)
        self.perplexities.append(perplexity)

        print(f'验证集 - Epoch {epoch}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}')
        return avg_loss, perplexity

    def train(self, num_epochs=50):
        """完整训练流程"""
        print("开始训练...")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, perplexity = self.validate(epoch)
            self.scheduler.step()

            print(f'Epoch {epoch}: 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, '
                  f'困惑度: {perplexity:.2f}, LR: {self.scheduler.get_last_lr()[0]:.6f}')

            if epoch % 10 == 0:
                self.save_model(f'model_epoch_{epoch}.pth')

        self.plot_training_curves()

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'perplexities': self.perplexities
        }, path)
        print(f"模型已保存到 {path}")

    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.perplexities)
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('验证集困惑度')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()


def generate_text_example(model, vocab, device, prompt="The history", max_length=50):
    """文本生成示例"""
    model.eval()

    # 将提示文本转换为token ID
    tokens = prompt.lower().split()
    input_ids = [vocab.index('<sos>')] if '<sos>' in vocab else [1]

    for token in tokens:
        if token in vocab:
            input_ids.append(vocab.index(token))
        else:
            input_ids.append(vocab.index('<unk>') if '<unk>' in vocab else 3)

    input_tensor = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        for i in range(max_length):
            # 创建掩码
            seq_len = input_tensor.size(1)
            mask = create_padding_mask(input_tensor).to(device)

            # 前向传播
            output_dict = model(input_tensor, mask)
            output = output_dict['output']

            # 获取下一个token的预测
            next_token_logits = output[:, -1, :]
            next_token = torch.softmax(next_token_logits, dim=-1).argmax(-1).item()

            # 检查是否生成结束标记
            eos_token = vocab.index('<eos>') if '<eos>' in vocab else 2
            if next_token == eos_token:
                break

            # 将新token添加到序列中
            input_ids.append(next_token)
            input_tensor = torch.tensor([input_ids]).to(device)

    # 将token ID转换回文本
    generated_tokens = []
    for idx in input_ids:
        if idx < len(vocab):
            token = vocab[idx]
            if token not in ['<sos>', '<pad>']:  # 过滤特殊token
                generated_tokens.append(token)

    generated_text = ' '.join(generated_tokens)
    print(f"生成文本: {generated_text}")
    return generated_text


def ablation_study():
    """消融实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 实验配置
    experiments = {
        'baseline': {'d_model': 256, 'n_heads': 4, 'num_layers': 4},
        'small_model': {'d_model': 128, 'n_heads': 2, 'num_layers': 2},
        'large_model': {'d_model': 512, 'n_heads': 8, 'num_layers': 6},
    }

    results = {}

    # 使用更小的配置以加快实验
    train_loader, val_loader, _, vocab = create_data_loaders(
        batch_size=8, seq_length=64, vocab_size=2000
    )
    vocab_size = len(vocab)

    for exp_name, config in experiments.items():
        print(f"\n=== 运行实验: {exp_name} ===")

        try:
            model = EncoderOnlyTransformer(
                vocab_size=vocab_size,
                num_classes=vocab_size,  # 语言建模任务，输出维度等于词表大小
                task_type='sequence_labeling',  # 序列标注任务（每个位置预测一个词）
                **config
            )
            trainer = TransformerTrainer(model, train_loader, val_loader, vocab_size, device)
            final_perplexity = quick_train_eval(trainer, epochs=3)  # 减少epoch数以加快实验
            results[exp_name] = final_perplexity
            print(f"{exp_name} 最终困惑度: {final_perplexity:.2f}")
        except Exception as e:
            print(f"实验 {exp_name} 失败: {e}")
            results[exp_name] = float('inf')

    plot_ablation_results(results)


def quick_train_eval(trainer, epochs=3):
    """快速训练和评估"""
    best_perplexity = float('inf')
    for epoch in range(1, epochs + 1):
        trainer.train_epoch(epoch)
        _, perplexity = trainer.validate(epoch)
        if perplexity < best_perplexity:
            best_perplexity = perplexity
    return best_perplexity


def plot_ablation_results(results):
    """绘制消融实验结果"""
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    values = [v if v != float('inf') else 1000 for v in results.values()]  # 处理无穷大值

    bars = plt.bar(names, values, color=['blue', 'green', 'orange'])
    plt.ylabel('困惑度 (越低越好)')
    plt.title('不同架构消融实验结果')
    plt.xticks(rotation=45)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                 f'{value:.2f}' if value < 1000 else '失败',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('ablation_results.png')
    plt.show()


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 超参数
    batch_size = 16
    seq_length = 64
    vocab_size = 10000
    num_epochs = 20
    data_dir = "data/wikitext-2-raw"  # WikiText-2数据目录

    # 创建数据加载器
    print("创建WikiText-2数据加载器...")
    try:
        train_loader, val_loader, test_loader, vocab = create_data_loaders(
            batch_size=batch_size,
            seq_length=seq_length,
            vocab_size=vocab_size,
            data_dir=data_dir
        )
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请检查数据文件路径是否正确")
        return

    # 创建编码器模型（用于语言建模）
    print("初始化编码器模型...")
    model = EncoderOnlyTransformer(
        vocab_size=len(vocab),
        num_classes=len(vocab),  # 语言建模任务，输出维度等于词表大小
        d_model=256,
        n_heads=4,
        num_layers=4,
        d_ff=512,
        dropout=0.3,
        max_len=seq_length,
        task_type='sequence_labeling'  # 序列标注任务（每个位置预测一个词）
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建训练器
    trainer = TransformerTrainer(model, train_loader, val_loader, len(vocab), device)

    try:
        # 开始训练
        trainer.train(num_epochs=num_epochs)

        # 文本生成示例
        print("\n文本生成示例:")
        generate_text_example(model, vocab, device, "The history of")

        # 运行消融实验
        print("\n开始消融实验...")
        ablation_study()

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    main()