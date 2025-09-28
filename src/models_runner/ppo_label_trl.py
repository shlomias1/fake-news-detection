from __future__ import annotations
import os, json, random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from utils.logger import _create_log

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, GenerationConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

# =========================
#   קונפיגים ותועלות
# =========================
@dataclass
class RLRewardCosts:
    r_correct: float = 1.0
    c_fp: float = 4.0     # false positive (REAL שסווג כ-FAKE)
    c_fn: float = 2.0     # false negative (FAKE שסווג כ-REAL)
    gamma_margin: float = 0.2  # בונוס ביטחון לפי margin

@dataclass
class PPORunCfg:
    base_model: str = "gpt2-medium"   # אם חסר זיכרון: "distilgpt2"
    use_8bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 1e-5
    batch_size: int = 8
    mini_batch_size: int = 4
    ppo_epochs: int = 4
    target_kl: float = 0.1
    max_prompt_len: int = 768
    max_new_tokens: int = 1          # מייצרים טוקן יחיד (תגית)
    temperature: float = 0.9
    top_p: float = 0.9
    seed: int = 42

# מיפוי הלייבלים אצלך
FAKE_LABEL_DATASET = 0  # {'fake': 0, 'real': 1}
REAL_LABEL_DATASET = 1

LABEL_REAL = "<REAL>"
LABEL_FAKE = "<FAKE>"

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# =========================
#     טעינת דאטה
# =========================
def load_df_texts_labels(df_path: str) -> Tuple[List[str], np.ndarray, pl.DataFrame]:
    df = pl.read_parquet(df_path) if str(df_path).endswith(".parquet") else pl.read_csv(df_path)
    # טקסט מועדף: titleplus_text_ns → title_ns_text → text_ns_text → title + text
    for c in ("titleplus_text_ns", "title_ns_text", "text_ns_text"):
        if c in df.columns:
            texts = df[c].cast(pl.Utf8).fill_null("").to_list()
            break
    else:
        t = df.get_column("title").fill_null("").cast(pl.Utf8) if "title" in df.columns else pl.Series([""]*df.height)
        x = df.get_column("text").fill_null("").cast(pl.Utf8)  if "text"  in df.columns else pl.Series([""]*df.height)
        texts = (t + pl.lit(" [SEP] ") + x).to_list()
    y = df.get_column("label").cast(pl.Int64).to_numpy()  # 0=fake, 1=real
    return texts, y, df

def build_prompts(texts: List[str]) -> List[str]:
    prompts = []
    for tx in texts:
        p = (
            "[INSTRUCTION] Decide if the news is FAKE or REAL.\n"
            f"Text: {tx}\n"
            "Label:"
        )
        prompts.append(p)
    return prompts

class NewsLabelDataset(Dataset):
    def __init__(self, prompts: List[str], labels_for_ppo: np.ndarray):
        # labels_for_ppo: 1=FAKE, 0=REAL (מותאם ל-reward)
        self.prompts = prompts
        self.labels = labels_for_ppo.astype(np.int64)
    def __len__(self): return len(self.prompts)
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "label": int(self.labels[idx])}

def split_train_val(prompts: List[str], y: np.ndarray, val_frac: float=0.2, seed: int=42):
    n = len(prompts)
    idx = np.arange(n); rng = np.random.RandomState(seed); rng.shuffle(idx)
    cut = int(n*(1.0-val_frac))
    tr, va = idx[:cut], idx[cut:]
    return ([prompts[i] for i in tr], y[tr]), ([prompts[i] for i in va], y[va])

# =========================
#   טוקנייזר ומודל (TRL)
# =========================
def make_tokenizer_and_model(cfg: PPORunCfg, device_map=None):
    import importlib
    from transformers import BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    added = tok.add_special_tokens({"additional_special_tokens": [LABEL_REAL, LABEL_FAKE]})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    has_cuda = torch.cuda.is_available()
    has_bnb  = importlib.util.find_spec("bitsandbytes") is not None

    # לא להשתמש ב-device_map="auto" כדי למנוע offloading
    load_kwargs = {"torch_dtype": torch.float32}

    # 8bit רק אם יש GPU + bitsandbytes; אחרת נבטל 8bit
    if cfg.use_8bit and has_cuda and has_bnb:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        cfg.use_8bit = False

    # טען את המודל (ללא offloading)
    model_vh = AutoModelForCausalLMWithValueHead.from_pretrained(cfg.base_model, **load_kwargs)

    # עדכון embedding אם הוספנו טוקנים
    if added > 0:
        model_vh.pretrained_model.resize_token_embeddings(len(tok))

    # LoRA
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM"
    )
    base = model_vh.pretrained_model
    if cfg.use_8bit:
        base = prepare_model_for_kbit_training(base)
    model_vh.pretrained_model = get_peft_model(base, lora)

    # העבר התקן ידנית (GPU אם קיים, אחרת CPU)
    device = torch.device("cuda") if has_cuda else torch.device("cpu")
    model_vh.to(device)

    id_fake = tok.encode(LABEL_FAKE, add_special_tokens=False)[0]
    id_real = tok.encode(LABEL_REAL, add_special_tokens=False)[0]
    return tok, model_vh, id_fake, id_real


# =========================
#       SFT קצר (אופציונלי)
# =========================
def run_sft_optional(tok, model_causal_with_value_head, train_prompts: List[str], y_for_ppo: np.ndarray,
                     epochs=1, lr=5e-5, bs=8):
    """
    SFT קצר כדי ללמד את המודל לענות בתגית. y_for_ppo: 1=FAKE, 0=REAL.
    """
    base = model_causal_with_value_head.pretrained_model
    class SFTDataset(Dataset):
        def __init__(self, prompts, y):
            self.prompts = prompts; self.y = y.astype(int)
        def __len__(self): return len(self.prompts)
        def __getitem__(self, i):
            label_text = LABEL_FAKE if self.y[i]==1 else LABEL_REAL
            text = self.prompts[i] + " " + label_text
            enc = tok(text, truncation=True, max_length=1024)
            return {"input_ids": torch.tensor(enc["input_ids"]), "attention_mask": torch.tensor(enc["attention_mask"])}
    ds = SFTDataset(train_prompts, y_for_ppo)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    args = TrainingArguments(
        output_dir="sft_tmp",
        per_device_train_batch_size=bs,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
    )
    tr = Trainer(model=base, args=args, train_dataset=ds, data_collator=collator)
    tr.train()

# =========================
#      פונקציית תגמול
# =========================
def compute_rewards_from_scores(
    scores: torch.Tensor,        # logits: [B, V] עבור הצעד שנוצר
    generated_ids: torch.Tensor, # ids שנוצרו: [B, 1] או [B]
    gold: torch.Tensor,          # 1=FAKE, 0=REAL
    id_fake: int, id_real: int,
    costs: RLRewardCosts
) -> torch.Tensor:
    probs = torch.softmax(scores, dim=-1)      # [B, V]
    p_fake = probs[:, id_fake]
    p_real = probs[:, id_real]
    out = generated_ids.squeeze(-1) if generated_ids.ndim==2 else generated_ids
    pred_is_fake = (out == id_fake)
    pred_is_real = (out == id_real)

    correct = (pred_is_fake & (gold==1)) | (pred_is_real & (gold==0))
    fp = (pred_is_fake & (gold==0))
    fn = (pred_is_real & (gold==1))

    base = torch.zeros_like(p_fake, dtype=torch.float32)
    base = base + costs.r_correct * correct.float()
    base = base - costs.c_fp * fp.float()
    base = base - costs.c_fn * fn.float()

    # בונוס ביטחון לפי מרווח בין שתי התגיות
    margin = (p_fake - p_real).abs()
    reward = base + costs.gamma_margin * margin
    return reward

# =========================
#        אימון PPO
# =========================
def ppo_train(
    df_feat_path: str,
    out_dir: str | Path,
    costs: RLRewardCosts = RLRewardCosts(),
    ppo_cfg: PPORunCfg = PPORunCfg(),
    val_frac: float = 0.2,
    do_sft_warmup: bool = True,
):
    set_seed(ppo_cfg.seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- דאטה ---
    texts, y_orig, _ = load_df_texts_labels(df_feat_path)   # y_orig: 0=fake, 1=real
    prompts = build_prompts(texts)
    (tr_prompts, y_tr_orig), (va_prompts, y_va_orig) = split_train_val(prompts, y_orig, val_frac, ppo_cfg.seed)

    # מיפוי ליעד PPO: 1=FAKE, 0=REAL
    y_tr = (y_tr_orig == FAKE_LABEL_DATASET).astype(np.int64)
    y_va = (y_va_orig == FAKE_LABEL_DATASET).astype(np.int64)

    # --- טוקנייזר/מודל ---
    tok, model_vh, id_fake, id_real = make_tokenizer_and_model(ppo_cfg, device_map="auto")

    # --- SFT (חימום) ---
    if do_sft_warmup:
        run_sft_optional(tok, model_vh, tr_prompts, y_tr, epochs=1, lr=5e-5, bs=max(2, ppo_cfg.mini_batch_size))

    # --- PPO ---
    ppo_config = PPOConfig(
        learning_rate=ppo_cfg.lr,
        batch_size=ppo_cfg.batch_size,
        mini_batch_size=ppo_cfg.mini_batch_size,
        ppo_epochs=ppo_cfg.ppo_epochs,
        target_kl=ppo_cfg.target_kl,
        seed=ppo_cfg.seed,
        log_with=None,
    )
    ppo_trainer = PPOTrainer(ppo_config, model=model_vh, ref_model=None, tokenizer=tok)

    train_ds = NewsLabelDataset(tr_prompts, y_tr)
    loader = DataLoader(train_ds, batch_size=ppo_cfg.batch_size, shuffle=True)

    gen_cfg = GenerationConfig(
        max_new_tokens=ppo_cfg.max_new_tokens,
        temperature=ppo_cfg.temperature,
        top_p=ppo_cfg.top_p,
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    model_vh.train()
    global_step = 0
    for epoch in range(2):  # התחל ב-2–3 אפוקים, תעלה אחרי פיילוט
        for batch in loader:
            global_step += 1
            batch_prompts = batch["prompt"]
            gold = torch.tensor(batch["label"], dtype=torch.long, device=ppo_trainer.accelerator.device)

            enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=ppo_cfg.max_prompt_len).to(ppo_trainer.accelerator.device)

            with torch.no_grad():
                gen_out = ppo_trainer.model.pretrained_model.generate(**enc, generation_config=gen_cfg)
            responses = gen_out.sequences[:, enc["input_ids"].shape[1]:]  # רק מה שנוצר
            scores_last = gen_out.scores[-1]  # [B, V]

            rewards = compute_rewards_from_scores(scores_last, responses, gold, id_fake, id_real, costs)

            # TRL מצפה לרשימות טנזורים ו-rewards כ-list של floats
            q_tensors = [t for t in enc["input_ids"]]
            r_tensors = [t for t in responses]
            rewards_list = rewards.detach().cpu().tolist()

            stats = ppo_trainer.step(q_tensors, r_tensors, rewards_list)
            if global_step % 20 == 0:
                # התאמה לגרסאות שונות של TRL: נסה כמה מפתחות
                kl = None
                for k in ("kl", "objective/kl", "ppo/kl"):
                    if k in stats:
                        kl = stats[k]; break
                mean_r = float(np.mean(rewards_list))
                print(f"[PPO] epoch={epoch+1} step={global_step} reward_mean={mean_r:.4f}  kl={kl}")

    # --- הערכה קצרה על הולידציה ---
    model_vh.eval()
    with torch.no_grad():
        enc_va = tok(va_prompts, return_tensors="pt", padding=True, truncation=True,
                     max_length=ppo_cfg.max_prompt_len).to(ppo_trainer.accelerator.device)
        gen_out = ppo_trainer.model.pretrained_model.generate(**enc_va, generation_config=GenerationConfig(
            max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores=True,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id))
        logits = gen_out.scores[-1]  # [B, V]
        probs = torch.softmax(logits, dim=-1)
        p_fake = probs[:, id_fake].detach().cpu().numpy()

    # --- שמירת ארטיפקטים ---
    (out_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out_dir / "tokenizer")
    ppo_trainer.model.save_pretrained(out_dir / "ppo_model")  # כולל value head + LoRA

    meta = {
        "label_mapping": {"fake": 0, "real": 1},
        "special_tokens": {"FAKE": LABEL_FAKE, "REAL": LABEL_REAL},
        "base_model": ppo_cfg.base_model,
        "use_8bit": ppo_cfg.use_8bit,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    eval_json = {
        "n_val": int(len(y_va_orig)),
        "p_fake_mean": float(np.mean(p_fake)),
        "p_fake_std": float(np.std(p_fake)),
        # לפי המיפוי המקורי:
        "label_counts": {"fake": int((y_va_orig==FAKE_LABEL_DATASET).sum()),
                         "real": int((y_va_orig==REAL_LABEL_DATASET).sum())}
    }
    with open(out_dir / "eval_val.json", "w", encoding="utf-8") as f:
        json.dump(eval_json, f, ensure_ascii=False, indent=2)

    print(f"[PPO] Saved artifacts to: {out_dir}")
    return {"out_dir": str(out_dir), "eval": eval_json}

# =========================
#        אינפרנס
# =========================
def ppo_predict_label(artifacts_dir: str | Path, title: str, text: str,
                      tau_low=0.3, tau_high=0.7) -> Dict[str, float | str]:
    """
    טוען את המודל המאומן ומחזיר תגית + הסתברות לפייק.
    """
    artifacts_dir = Path(artifacts_dir)
    tok = AutoTokenizer.from_pretrained(artifacts_dir / "tokenizer")
    model_vh = AutoModelForCausalLMWithValueHead.from_pretrained(artifacts_dir / "ppo_model", device_map="auto")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    id_fake = tok.encode(LABEL_FAKE, add_special_tokens=False)[0]
    id_real = tok.encode(LABEL_REAL, add_special_tokens=False)[0]

    prompt = (
        "[INSTRUCTION] Decide if the news is FAKE or REAL.\n"
        f"Title: {title or ''}\n"
        f"Text: {text or ''}\n"
        "Label:"
    )
    with torch.no_grad():
        enc = tok([prompt], return_tensors="pt", padding=True, truncation=True, max_length=768).to(model_vh.pretrained_model.device)
        gen = model_vh.pretrained_model.generate(**enc, max_new_tokens=1, do_sample=False,
                                                 return_dict_in_generate=True, output_scores=True,
                                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        logits = gen.scores[-1]  # [1, V]
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        p_fake = float(probs[id_fake].item())

    if p_fake < tau_low:
        return {"label": "REAL", "prob_fake": p_fake}
    elif p_fake > tau_high:
        return {"label": "FAKE", "prob_fake": p_fake}
    else:
        return {"label": "ABSTAIN", "prob_fake": p_fake}