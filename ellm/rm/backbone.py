import json
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from llm_blender.pair_ranker.config import RankerConfig
from llm_blender.pair_ranker.model_util import build_collator, build_tokenizer
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Model, DebertaV2PreTrainedModel, SequenceClassifierOutput)
from transformers.utils.hub import TRANSFORMERS_CACHE


class DebertaV2PairRM(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.n_tasks = config.n_tasks
        self.drop_out = config.drop_out

        # LM
        self.pretrained_model = DebertaV2Model(config)
        self.hidden_size = config.hidden_size

        self.sep_token_id = config.sep_token_id  # to add
        self.source_prefix_id = config.source_prefix_id  # to add
        self.cand_prefix_id = config.cand_prefix_id
        # self.cand1_prefix_id = config.cand1_prefix_id
        # self.cand2_prefix_id = config.cand2_prefix_id

        # self.head_layer = nn.Sequential(
        #     nn.Dropout(self.drop_out),
        #     nn.Linear(2 * self.hidden_size, 1 * self.hidden_size),
        #     nn.Tanh(),
        #     nn.Dropout(self.drop_out),
        #     nn.Linear(1 * self.hidden_size, self.n_tasks),
        # )
        # self.sigmoid = nn.Sigmoid()

        # Initialize weights and apply final processing
        self.post_init()
        self.eval()
        self.prepare_ranker("llm-blender/PairRM")

    @torch.no_grad
    def get_feature(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        """Get the feature \phi(s, a) in a singleton form."""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        #  <source_prefix_id>...<sep><cand_prefix_id>...<sep>
        assert all(
            [self.source_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]
        ), "<source> id not in input_ids"
        assert all(
            [self.cand_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]
        ), "<candidate> id not in input_ids"

        keep_column_mask = attention_mask.ne(0).any(dim=0)
        input_ids = input_ids[:, keep_column_mask]
        attention_mask = attention_mask[:, keep_column_mask]
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )
        encs = outputs.hidden_states[-1]
        source_idxs = torch.where(input_ids == self.source_prefix_id)
        source_encs = encs[source_idxs[0], source_idxs[1], :]
        cand_idxs = torch.where(input_ids == self.cand_prefix_id)
        cand_encs = encs[cand_idxs[0], cand_idxs[1], :]

        # reduce
        source_cand_encs = torch.cat([source_encs, cand_encs], dim=-1)
        return source_cand_encs.detach()

    def prepare_ranker(self, ranker_path, **kwargs):
        cache_dir = kwargs.pop("cache_dir", TRANSFORMERS_CACHE)

        ranker_path = os.path.join(cache_dir, ranker_path)
        ranker_path = Path(ranker_path)
        with open(ranker_path / "config.json", "r") as f:
            ranker_config_json = json.load(f)
        ranker_config = RankerConfig.from_dict(ranker_config_json)

        self.tokenizer = build_tokenizer(
            ranker_config.model_name, cache_dir=ranker_config.cache_dir
        )
        self.ranker_collator = build_collator(
            ranker_config.ranker_type,
            self.tokenizer,
            ranker_config.source_maxlength,
            ranker_config.candidate_maxlength,
        )

    # def compare(
    #     self,
    #     inputs: List[str],
    #     candidates_A: List[str],
    #     candidates_B: List[str],
    #     instructions: List[str] = None,
    #     batch_size: int = 4,
    #     return_logits: bool = False,
    #     mode: str = "[A,B]+[B,A]",
    #     disable_tqdm: bool = False,
    # ):
    #     assert len(candidates_A) == len(
    #         candidates_B
    #     ), "Number of candidates_A and candidates_B must be the same"
    #     assert len(inputs) == len(
    #         candidates_A
    #     ), "Number of inputs and candidates must be the same"
    #     candidates = [[a, b] for a, b in zip(candidates_A, candidates_B)]

    #     if mode in ["[A,B]", "[B,A]"]:
    #         if mode == "[B,A]":
    #             candidates = [[b, a] for a, b in zip(candidates_A, candidates_B)]
    #         collate_fn = copy.copy(self.ranker_collator)
    #         dataset = RankerDataset(inputs, candidates, instructions=instructions)
    #         dataloader = torch.utils.data.DataLoader(
    #             dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    #         )
    #         cmp_results = []
    #         with torch.no_grad():
    #             for batch in tqdm(
    #                 iter(dataloader),
    #                 desc="Ranking candidates",
    #                 disable=disable_tqdm,
    #             ):
    #                 batch = {k: v.to("cuda") for k, v in batch.items() if v is not None}
    #                 source_ids, source_attention_mask = (
    #                     batch["source_ids"],
    #                     batch["source_attention_mask"],
    #                 )
    #                 left_cand_ids, left_cand_attention_mask = (
    #                     batch["candidate_ids"][:, 0],
    #                     batch["candidate_attention_mask"][:, 0],
    #                 )
    #                 right_cand_ids, right_cand_attention_mask = (
    #                     batch["candidate_ids"][:, 1],
    #                     batch["candidate_attention_mask"][:, 1],
    #                 )
    #                 import pdb

    #                 pdb.set_trace()

    #                 outputs = self._forward(
    #                     source_ids,
    #                     source_attention_mask,
    #                     left_cand_ids,
    #                     left_cand_attention_mask,
    #                     right_cand_ids,
    #                     right_cand_attention_mask,
    #                     left_scores,
    #                     right_scores,
    #                 )
    #                 cmp_results.append(outputs["logits"].detach().cpu().numpy())
    #         cmp_results = np.concatenate(cmp_results, axis=0)
    #     else:
    #         raise NotImplementedError
    #     if return_logits:
    #         return cmp_results
    #     else:
    #         return cmp_results > 0

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        #  <source_prefix_id>...<sep><cand1_prefix_id>...<sep><cand2_prefix_id> ... <sep>
        assert all(
            [self.source_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]
        ), "<source> id not in input_ids"
        assert all(
            [self.cand1_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]
        ), "<candidate1> id not in input_ids"
        assert all(
            [self.cand2_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]
        ), "<candidate2> id not in input_ids"

        keep_column_mask = attention_mask.ne(0).any(dim=0)
        input_ids = input_ids[:, keep_column_mask]
        attention_mask = attention_mask[:, keep_column_mask]
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )
        encs = outputs.hidden_states[-1]
        source_idxs = torch.where(input_ids == self.source_prefix_id)
        source_encs = encs[source_idxs[0], source_idxs[1], :]
        cand1_idxs = torch.where(input_ids == self.cand1_prefix_id)
        cand1_encs = encs[cand1_idxs[0], cand1_idxs[1], :]
        cand2_idxs = torch.where(input_ids == self.cand2_prefix_id)
        cand2_encs = encs[cand2_idxs[0], cand2_idxs[1], :]

        # reduce
        source_cand1_encs = torch.cat([source_encs, cand1_encs], dim=-1)
        source_cand2_encs = torch.cat([source_encs, cand2_encs], dim=-1)
        left_pred_scores = self.head_layer(source_cand1_encs)
        right_pred_scores = self.head_layer(source_cand2_encs)

        loss = None
        if labels is not None:
            loss = self.compute_loss(left_pred_scores, right_pred_scores, labels)

        preds = (left_pred_scores - right_pred_scores).mean(dim=-1)
        return SequenceClassifierOutput(
            loss=loss,
            logits=preds,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )

    def compute_loss(self, left_pred_scores, right_pred_scores, labels):
        """
        Args:
            left_pred_scores: [n_candidates, n_task]
            right_pred_scores: [n_candidates, n_task]
            labels: [n_candidates, n_task], 1/0/-1 for left/right/both is better
        """

        device = left_pred_scores.device
        loss = torch.tensor(0.0).to(left_pred_scores.device)

        dif_scores = labels
        left_pred_scores = left_pred_scores * dif_scores.sign()
        right_pred_scores = -right_pred_scores * dif_scores.sign()
        cls_loss = torch.tensor(0.0, device=device)
        cls_loss += -torch.log(
            torch.sigmoid(left_pred_scores + right_pred_scores)
        ).mean()
        loss += cls_loss
        return loss
