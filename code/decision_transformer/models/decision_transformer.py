import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


from kmeans_pytorch import kmeans

import os

class ModelSettings:
    embd_pdrop = 0.0
    resid_pdrop = 0.0
    attn_pdrop = 0.0

    def __init__(
        self, size: str, n_layer: int, d_model: int, learning_rate: float,
    ):
        self.size = size
        self.n_layer = n_layer
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.n_head = max(2, self.d_model // 64)
        self.d_ff = 4 * d_model
        self.d_attn = 1 * d_model


# hparams from Scaling Laws for Neural Languages
common_models_by_name = {
    "x10small": ModelSettings(
        size="x10small", n_layer=1, d_model=8, learning_rate=0.00211,
    ),
    "x9small": ModelSettings(
        size="x9small", n_layer=1, d_model=16, learning_rate=0.00211,
    ),
    "x8small": ModelSettings(
        size="x8small", n_layer=1, d_model=32, learning_rate=0.00211,
    ),
    "x7small": ModelSettings(
        size="x7small", n_layer=2, d_model=32, learning_rate=0.00211,
    ),
    "x6small": ModelSettings(
        size="x6small", n_layer=2, d_model=64, learning_rate=0.00211,
    ),
    "x5small": ModelSettings(
        size="x5small", n_layer=2, d_model=128, learning_rate=0.00202,
    ),
    "x4small": ModelSettings(
        size="x4small", n_layer=4, d_model=256, learning_rate=0.00173,
    ),
    "x3small": ModelSettings(
        size="x3small", n_layer=4, d_model=512, learning_rate=0.00163,
    ),
    "x2small": ModelSettings(
        size="x2small", n_layer=8, d_model=512, learning_rate=0.00144,
    ),
    "x1small": ModelSettings(
        size="x1small", n_layer=6, d_model=768, learning_rate=0.00146,
    ),
    "small": ModelSettings(
        size="small", n_layer=12, d_model=768, learning_rate=0.0006,
    ),
    "medium": ModelSettings(
        size="medium", n_layer=24, d_model=1024, learning_rate=0.0003,
    ),
    "large": ModelSettings(
        size="large", n_layer=24, d_model=1536, learning_rate=0.00025,
    ),
    "xl": ModelSettings(size="xl", n_layer=24, d_model=2048, learning_rate=0.00000625,),
}


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        args,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        print(args)
        print("model config of size {}".format(args["pretrained_lm"]))
        if not args["scratch"]:
            print("Loading from pretrained")
            if args["pretrained_lm"]=="gpt2":
                config = transformers.GPT2Config.from_pretrained(args["pretrained_lm"])
            
                config.attn_pdrop = 0.1  # args["dropout"]
                config.resid_pdrop = args["dropout"]
                self.transformer = GPT2Model.from_pretrained(
                    args["pretrained_lm"],
                    config=config,
                )
            else:

                config = common_models_by_name.get(args["pretrained_lm"])
                n_layer = config.n_layer
                d_model = config.d_model
                n_head = config.n_head
                d_ff = config.d_ff
                d_attn = config.d_attn
                d_embd = d_model

                config = transformers.GPT2Config(
                    vocab_size=50257,
                    n_ctx=1024,
                    n_positions=1024,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_inner=d_ff,
                    n_embd=d_embd,
                    bos_token_id=50256,
                    eos_token_id=50256,
                    attn_pdrop=0.0,
                    embd_pdrop=0.0,
                    resid_pdrop=0.0,
                )
                checkpoint_file = "{}/{}.pt".format(args["checkpoints_dir"], args["pretrained_lm"])
                checkpoint = torch.load(
                    checkpoint_file, map_location="cuda"
                )
                state_dict = checkpoint["model_state_dict"]
                self.transformer = GPT2Model.from_pretrained(
                    state_dict=state_dict,
                    config=config,
                    pretrained_model_name_or_path=None
                )
            print("Parameters", sum(p.numel() for p in self.transformer.parameters()))
            
            if args["gpt_kmeans"] is not None:
                if args["kmeans_cache"] is not None and not os.path.exists(
                    args["kmeans_cache"]
                ):
                    cluster_ids_x, self.cluster_centers = kmeans(
                        X=self.transformer.wte.weight.data,
                        num_clusters=args["gpt_kmeans"],
                        distance="cosine",
                        device=args.get("device", "cuda"),
                    )
                    if args["kmeans_cache"] is not None:
                        torch.save(self.cluster_centers, args["kmeans_cache"])
                else:
                    self.cluster_centers = torch.load(args["kmeans_cache"])
                self.cluster_centers = self.cluster_centers.to(
                    args.get("device", "cuda")
                )
                # self.cluster_centers.requires_grad = True
            hidden_size = config.n_embd
            self.hidden_size = config.n_embd

        else:
            print("Training from scratch")
            if args["pretrained_lm"]=="gpt2":
                config = transformers.GPT2Config.from_pretrained(args["pretrained_lm"])
                self.transformer = GPT2Model(config)
            else:

                config = common_models_by_name.get(args["pretrained_lm"])
                n_layer = config.n_layer
                d_model = config.d_model
                n_head = config.n_head
                d_ff = config.d_ff
                d_attn = config.d_attn
                d_embd = d_model

                config = transformers.GPT2Config(
                    vocab_size=50257,
                    n_ctx=1024,
                    n_positions=1024,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_inner=d_ff,
                    n_embd=d_embd,
                    bos_token_id=50256,
                    eos_token_id=50256,
                    attn_pdrop=0.0,
                    embd_pdrop=0.0,
                    resid_pdrop=0.0,
                )
                self.transformer = GPT2Model(config)
            hidden_size = config.n_embd
            self.hidden_size = config.n_embd

            # config = transformers.GPT2Config(
            #     # vocab_size=1,  # doesn't matter -- we don't use the vocab
            #     n_embd=hidden_size,
            #     **kwargs,
            # )

            # self.transformer = GPT2Model(config)
        if max_ep_len > config.n_positions and args["extend_positions"]:
            current_max_pos, embed_size = self.transformer.wpe.weight.shape
            new_encoder_pos_embed = self.transformer.wpe.weight.new_empty(
                max_ep_len, embed_size
            )
            # copy position embeddings over and over to initialize the new position embeddings
            orig_k = 2
            k = orig_k
            step = current_max_pos - k
            new_encoder_pos_embed[:k] = self.transformer.wpe.weight[:k]
            while k < max_ep_len - 1:
                new_encoder_pos_embed[k : (k + step)] = self.transformer.wpe.weight[
                    orig_k : min(max_ep_len - k + orig_k, current_max_pos)
                ]
                k += step
            self.transformer.wpe.weight.data = new_encoder_pos_embed
        
        if args["frozen"]:
            if not args["freeze_schedule"]:
                self.freeze_params(args["frozen_uce_style"])
            else:
                print("Freezeing everything")
                for name, param in self.transformer.named_parameters():
                    param.requires_grad = False

        else:
            print("Not feezing")
            
        if args["extend_positions"]:
            self.embed_timestep = self.transformer.wpe
        else:
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        if args["share_input_output_proj"]:
            self.predict_state = lambda x: F.linear(x, self.embed_state.weight.t())
            self.predict_return = lambda x: F.linear(x, self.embed_return.weight.t())
            self.predict_action = lambda x: F.tanh(
                F.linear(x, self.embed_action.weight.t())
            )
        else:
            self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
            self.predict_action = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.act_dim)]
                    + ([nn.Tanh()] if action_tanh else [])
                )
            )
            self.predict_return = torch.nn.Linear(hidden_size, 1)

        self.past_key_values = None
        print(self)
    
    def freeze_params(self, frozen_uce_style):
        if frozen_uce_style:
            print("Partial freezeing UCE Style")
            # # freeze all parameters except the layernorm and positional embeddings
            for name, param in self.transformer.named_parameters():
                if 'ln' in name or 'wpe' in name:
                    print(name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            # print("Freezeing everything")
            # for name, param in self.transformer.named_parameters():
            #     param.requires_grad = False
        else:
            print("Freezing original style")
            for param in self.transformer.h.parameters():
                param.requires_grad = False
            for name, param in self.transformer.named_parameters():
                if param.requires_grad == True:
                    print(name)
    
    def unfreeze_transformer(self):
        print("Unfreezeing everything")
        for name, param in self.transformer.named_parameters():
            param.requires_grad = True

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask=None,
        past_key_values=None,
    ):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        # state_embeddings = state_embeddings + time_embeddings
        # action_embeddings = action_embeddings + time_embeddings
        # returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        all_embs = self.embed_ln(stacked_inputs)

        stacked_inputs = all_embs + time_embeddings.repeat_interleave(3, dim=1)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            past_key_values=None,  # self.past_key_values,
            use_cache=True,
        )
        x = transformer_outputs["last_hidden_state"]
        self.past_key_values = transformer_outputs["past_key_values"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(
            x[:, 2]
        )  # predict next return given state and action
        state_preds = self.predict_state(
            x[:, 2]
        )  # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state

        return state_preds, action_preds, return_preds, all_embs

    def get_action(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        past_key_values=None,
        **kwargs
    ):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            attention_mask = attention_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds, __ = self.forward(
            states,
            actions,
            None,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
            **kwargs,
        )

        return action_preds[0, -1]
