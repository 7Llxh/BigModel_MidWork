Transformer 消融实验



项目简介



基于PyTorch的Transformer组件消融实验，研究不同架构组件对模型性能的影响。



硬件要求



最低配置:

• GPU: ≥4GB VRAM (GTX 1650, RTX 3050)



• 内存: 8GB RAM



• 存储: 1GB可用空间



推荐配置:

• GPU: ≥8GB VRAM (RTX 3070, RTX 4070) 



• 内存: 16GB RAM



• 存储: 2GB可用空间



环境配置



\# 创建虚拟环境

python -m venv transformer-env

source transformer-env/bin/activate  # Linux/Mac

\# 或 transformer-env\\Scripts\\activate  # Windows



\# 安装PyTorch (选择适合您CUDA版本的命令)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8

\# 或 pip install torch torchvision torchaudio  # CPU版本



\# 安装依赖

pip install numpy matplotlib tqdm





数据准备



\# 下载并解压数据集

mkdir -p data/wikitext-2-raw

wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip -P data/

unzip data/wikitext-2-raw-v1.zip -d data/

rm data/wikitext-2-raw-v1.zip





运行实验



\# 运行完整消融实验（约1-2小时）

python transformer\_ablation.py



\# 快速测试（修改代码仅运行2个实验）

\# 在代码中将key\_experiments改为只包含"基线模型"和"1头注意力"

python transformer\_ablation.py





精确命令行示例



Linux/Ubuntu:

python3 -m venv ~/venv/transformer \&\& source ~/venv/transformer/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install numpy matplotlib tqdm

mkdir -p data/wikitext-2-raw

wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip -P data/ \&\& unzip data/wikitext-2-raw-v1.zip -d data/ \&\& rm data/wikitext-2-raw-v1.zip

python transformer\_ablation.py





Windows PowerShell:

python -m venv transformer-env; .\\transformer-env\\Scripts\\Activate.ps1

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install numpy matplotlib tqdm

mkdir data/wikitext-2-raw

\# 手动下载wikitext-2-raw-v1.zip并解压到data/wikitext-2-raw/

python transformer\_ablation.py





实验内容



• 基线Transformer模型



• 不同注意力头数比较（1/2/8头）



• 位置编码消融实验



• 残差连接消融实验



• 模型大小比较（小型/大型）



预期输出



• 训练/验证损失曲线



• 准确率对比图表



• 各组件影响分析报告



• 保存的模型文件（.pth格式）

