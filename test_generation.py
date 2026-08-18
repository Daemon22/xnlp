import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from core_llm.architecture import XNLPCoreLLM, XNLPConfig
from core_llm.tokenizer import XNLPTokenizer

tokenizer = XNLPTokenizer.load_pretrained('trained_model/tokenizer')
print(f'Tokenizer vocab: {tokenizer.vocab_size_actual}')

ckpt = torch.load('trained_model/best_model.pt', map_location='cpu', weights_only=False)
cfg = ckpt['config']
config = XNLPConfig(
    vocab_size=cfg['vocab_size'],
    hidden_size=cfg['hidden_size'],
    intermediate_size=cfg['intermediate_size'],
    num_hidden_layers=cfg['num_layers'],
    num_attention_heads=cfg['num_heads'],
    num_key_value_heads=cfg['num_kv_heads'],
    max_position_embeddings=cfg['max_seq_len'],
    device='cpu',
)
model = XNLPCoreLLM(config)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f'Model params: {total_params:,}')
print(f'Trained for: {ckpt["epoch"]} epochs')
print(f'Final loss: {ckpt["loss"]:.4f}')

prompts = [
    'Umntu ngumntu',
    'Imbongi yethu',
    'Ityala lamawele',
    'UMqhayi',
    'Isibongo',
    'Nkosi sikelela',
    'Indoda ngumuntu',
    'Ukuthetha iqiniso',
]

print('\n=== Generation Test ===')
for p in prompts:
    ids = tokenizer.encode(p, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=30, temperature=0.7, top_k=40, top_p=0.9)
    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    print(f'  {p:25s} -> {text}')
