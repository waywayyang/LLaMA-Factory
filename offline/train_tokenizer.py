from llamafactory.model.way.configuration import WayConfig
from llamafactory.model.way.model import WayModel
from llamafactory.model.way.tokenizer import WayTokenizerFast
from tokenizers import ByteLevelBPETokenizer
file_paths=["/root/data/pretrain_sanguoyanyi.txt"]
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=file_paths, vocab_size=50000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
tokenizer = WayTokenizerFast(tokenizer_object=tokenizer)
tokenizer.save_pretrained("/root/data/sanguo2")