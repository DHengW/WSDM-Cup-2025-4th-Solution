from dataclasses import dataclass
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.file_utils import ModelOutput
import torch.nn.functional as F


@dataclass
class RankerOutput(ModelOutput):
    distillation_loss: Optional[Tensor] = None
    ce_loss: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None


class EediRanker(nn.Module):
    def __init__(self, base_model, tokenizer):
        super().__init__()

        self.model = base_model
        self.config = self.model.config
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.divergence_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.cos_loss_fn = nn.CosineEmbeddingLoss()
        self.num_labels = 2
        self.token_ids = []

        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.tok_locations = []
        for letter in letters[: self.num_labels]:
            token_id = tokenizer(letter, add_special_tokens=False)["input_ids"][-1]
            self.tok_locations.append(token_id)

        for idx, letter in enumerate(letters[: self.num_labels]):
            print(f">> EediRanker: {letter} token id: {self.tok_locations[idx]}")

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def encode(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        scores = []
        for token_id in self.tok_locations: # Take the logits of "A" and "B"
            score = outputs.logits[:, -1, token_id]  
            scores.append(score)

        logits = torch.stack(scores, 1)  # [bs, 2]

        return logits.contiguous()

    def forward(self, input_ids, attention_mask, labels=None, grads=None, ce_mask=None, temperature=1.0, use_distillation=False, **kwargs):
        logits = self.encode(input_ids, attention_mask)

        loss = None
        distillation_loss = None
        ce_loss = None

        if labels is not None:
            labels = labels.to(logits.device).reshape(-1)  # bs
            ce_loss = (self.loss_fn(logits, labels)).mean()

            if use_distillation:
                if grads is not None:
                    grads1 = grads[:, :2]
                    grads2 = grads[:, 2:4]
                    grads3 = grads[:, 4:6]
                    loss_grad1 = self.divergence_loss_fn(
                        F.log_softmax(logits / temperature, dim=1),
                        F.softmax(grads1 / temperature, dim=1)
                    )
                    cos_loss1 = self.cos_loss_fn(F.softmax(grads1 / temperature, dim=1), 
                                                 F.softmax(logits / temperature, dim=1),
                                            torch.ones(logits.size()[0]).to(logits.device))
                    loss_grad2 = self.divergence_loss_fn(
                        F.log_softmax(logits / temperature, dim=1),
                        F.softmax(grads2 / temperature, dim=1)
                    )
                    cos_loss2 = self.cos_loss_fn(F.softmax(grads2 / temperature, dim=1), 
                                                 F.softmax(logits / temperature, dim=1),
                                            torch.ones(logits.size()[0]).to(logits.device))
                    loss_grad3 = self.divergence_loss_fn(
                        F.log_softmax(logits / temperature, dim=1),
                        F.softmax(grads3 / temperature, dim=1)
                    )
                    cos_loss3 = self.cos_loss_fn(F.softmax(grads3 / temperature, dim=1), 
                                                 F.softmax(logits / temperature, dim=1),
                                            torch.ones(logits.size()[0]).to(logits.device))
                    
                    loss = (ce_loss + loss_grad1 + cos_loss1 + loss_grad2 + cos_loss2 + loss_grad3 + cos_loss3) / 7. 
                else:
                    raise ValueError("Teacher logits are required for distillation loss")
            else:
                loss = ce_loss
                distillation_loss = torch.tensor(0.0, device=logits.device)

        return RankerOutput(loss=loss, logits=logits, distillation_loss=distillation_loss, ce_loss=ce_loss)

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)

class WSDMRanker(nn.Module):
    def __init__(self, base_model, tokenizer):
        super().__init__()

        self.model = base_model
        self.config = self.model.config
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.num_labels = 2
        self.token_ids = []

        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.tok_locations = []
        for letter in letters[: self.num_labels]:
            token_id = tokenizer(letter, add_special_tokens=False)["input_ids"][-1]
            self.tok_locations.append(token_id)

        for idx, letter in enumerate(letters[: self.num_labels]):
            print(f">> EediRanker: {letter} token id: {self.tok_locations[idx]}")

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def encode(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        scores = []
        for token_id in self.tok_locations:
            score = outputs.logits[:, -1, token_id]  # [bs]
            scores.append(score)

        logits = torch.stack(scores, 1)  # [bs, num_labels]
        return logits.contiguous(),outputs.logits[:, -1, :].contiguous()

    def forward(self, input_ids, attention_mask, labels=None, teacher_logits=None, ce_mask=None, temperature=1.0, use_distillation=False, **kwargs):
        logits,all_logits = self.encode(input_ids, attention_mask)

        loss = None
        distillation_loss = None
        ce_loss = None

        if labels is not None:
            labels = labels.to(logits.device).reshape(-1)  # bs
            # ce_mask = ce_mask.to(logits.device).reshape(-1)
            ce_loss = (self.loss_fn(all_logits, labels)).mean()

            if use_distillation:
                if teacher_logits is not None:
                    teacher_targets = teacher_logits  # .reshape(-1, self.group_size)
                    teacher_targets = torch.softmax(teacher_targets.detach() / temperature, dim=-1)
                    distillation_loss = -torch.mean(torch.sum(torch.log_softmax(logits, dim=-1) * teacher_targets, dim=-1))
                    loss = 0.5 * ce_loss + 0.5 * distillation_loss  # alpha = 0.5, beta = 0.5
                else:
                    raise ValueError("Teacher logits are required for distillation loss")
            else:
                loss = ce_loss
                distillation_loss = torch.tensor(0.0, device=logits.device)

        return RankerOutput(loss=loss, logits=logits, distillation_loss=distillation_loss, ce_loss=ce_loss)

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)