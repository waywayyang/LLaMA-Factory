from llamafactory.model.way.configuration import WayConfig
from llamafactory.model.way.model import WayModel
config={
    "vocab_size": 151643,
    "hidden_size": 512,
    "intermediate_size": 512,
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "hidden_act": "silu",
    "max_position_embeddings": 512,
    "initializer_range": 0.02,
    "rms_norm_eps": 0.000001,
    "use_cache": True,
    "tie_word_embeddings": False,
    "rope_theta": 10000,
    "rope_scaling": None,
    "use_sliding_window": False,
    "sliding_window": 131072,
    "max_window_layers": 28,
    "attention_dropout": 0
}
configuration = WayConfig(**config)
model = WayModel(configuration)
model.save_pretrained("/root/LLaMA-Factory/examples/train_pretrain/ckpt")
