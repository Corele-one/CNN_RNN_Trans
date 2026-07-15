import config
import torch
from torch.utils.data import DataLoader
import os
import glob

from tools.data_loader import MTDataset
from model.tf_model import make_model
import logging
import sacrebleu
from tqdm import tqdm

from beam_decoder import beam_search
from model.train_utils import MultiGPULossCompute, get_std_opt
from tools.tokenizer_utils import chinese_tokenizer_load
from tools.create_exp_folder import create_exp_folder


logging.basicConfig(format='%(asctime)s-%(name)s-%(levelname)s-%(message)s', level=logging.INFO)

def load_pretrained_model(model, model_path=None):
    """加载预训练模型"""
    if model_path is None:
        # 自动查找最新的最佳模型
        weights_dirs = glob.glob('run/train/*/weights')
        if weights_dirs:
            latest_dir = max(weights_dirs, key=os.path.getmtime)
            model_files = glob.glob(f'{latest_dir}/best_bleu_*.pth')
            if model_files:
                model_path = max(model_files, key=os.path.getmtime)  # 获取最新的最佳模型
    
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=config.device))
            logging.info(f"✅ 成功加载预训练模型: {model_path}")
            return True
        except Exception as e:
            logging.error(f"❌ 加载模型失败: {e}")
            return False
    else:
        logging.info("🆕 未找到预训练模型，将从头开始训练")
        return False

def run_epoch(data, model, loss_compute):
    total_tokens = 0.
    total_loss = 0.
    
    for batch in tqdm(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
    
    return total_loss / total_tokens

def train(train_data, dev_data, model, model_par, criterion, optimizer, start_epoch=1, resume_training=False):
    """训练并保存模型"""
    best_bleu_score = -float('inf')
    exp_folder, weights_folder = create_exp_folder()
    
    # 如果继续训练，尝试加载之前最好的BLEU分数
    if resume_training:
        # 查找之前的最佳模型以获取BLEU分数
        existing_models = glob.glob(f'{weights_folder}/best_bleu_*.pth')
        if existing_models:
            # 从文件名中提取BLEU分数
            import re
            bleu_scores = []
            for model_path in existing_models:
                match = re.search(r'best_bleu_(\d+\.\d+)\.pth', model_path)
                if match:
                    bleu_scores.append(float(match.group(1)))
            if bleu_scores:
                best_bleu_score = max(bleu_scores)
                logging.info(f"📊 恢复训练，之前的最佳BLEU分数: {best_bleu_score:.2f}")

    for epoch in range(start_epoch, config.epoch_num + 1):
        logging.info(f"第{epoch}轮模型训练与验证")
        
        model.train()
        train_loss = run_epoch(train_data, model_par,
                               MultiGPULossCompute(model.generator, criterion, config.device_id, optimizer))

        model.eval()
        dev_loss = run_epoch(dev_data, model_par,
                             MultiGPULossCompute(model.generator, criterion, config.device_id, None))

        bleu_score = evaluate(dev_data, model)
        logging.info(f"Epoch: {epoch}, train_loss: {train_loss:.3f}, val_loss: {dev_loss:.3f}, Bleu Score: {bleu_score:.2f}\n")

        if bleu_score > best_bleu_score:
            if best_bleu_score != -float('inf'):
                old_model_path = f"{weights_folder}/best_bleu_{best_bleu_score:.2f}.pth"
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)

            model_path_best = f"{weights_folder}/best_bleu_{bleu_score:.2f}.pth"
            torch.save(model.state_dict(), model_path_best)
            best_bleu_score = bleu_score
            logging.info(f"🏆 新的最佳模型保存: BLEU = {bleu_score:.2f}")

        if epoch == config.epoch_num:
            model_path_last = f"{weights_folder}/last_bleu_{bleu_score:.2f}.pth"
            torch.save(model.state_dict(), model_path_last)
            logging.info(f"📁 最终模型保存: BLEU = {bleu_score:.2f}")

def evaluate(data, model):
    """在data上用训练好的模型进行预测"""
    sp_chn = chinese_tokenizer_load()
    trg = []
    res = []
    
    with torch.no_grad():
        for batch in tqdm(data):
            cn_sent = batch.trg_text
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            
            decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                         config.padding_idx, config.bos_idx, config.eos_idx,
                                         config.beam_size, config.device)
            
            decode_result = [h[0] for h in decode_result]
            translation = [sp_chn.decode_ids(_s) for _s in decode_result]
            trg.extend(cn_sent)
            res.extend(translation)
    
    trg = [trg]
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')
    return float(bleu.score)

def run(resume_training=True, model_path=None):
    """主训练函数"""
    # 创建数据集
    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    # 初始化模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    
    model_par = torch.nn.DataParallel(model)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    optimizer = get_std_opt(model)

    # 尝试加载预训练模型
    loaded = False
    if resume_training:
        loaded = load_pretrained_model(model, model_path)
    
    start_epoch = 1
    if loaded:
        # 如果成功加载模型，可以选择从哪个epoch开始继续训练
        # 这里简单处理，可以从第1个epoch开始继续训练
        logging.info("🔄 将继续从已加载的模型权重进行训练")
    
    # 开始训练
    train(train_dataloader, dev_dataloader, model, model_par, criterion, optimizer, 
          start_epoch=start_epoch, resume_training=resume_training)
    
    # 可选：测试最终模型
    # test(test_dataloader, model, criterion)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import warnings
    warnings.filterwarnings('ignore')
    
    # 设置继续训练参数
    RESUME_TRAINING = True  # 设置为True以继续训练
    MODEL_PATH = None       # 设置为具体路径或None让程序自动查找
    
    run(resume_training=RESUME_TRAINING, model_path=MODEL_PATH)
