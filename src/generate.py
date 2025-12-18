import torch
import torch.nn.functional as F

@torch.no_grad()
def generate_text(model, output_layer, stoi, itos,
                  start_char="H", length=300, temperature=1.0, device="cpu"):
    model.eval()

    idx = torch.tensor([stoi[start_char]], device=device)
    x = F.one_hot(idx, len(stoi)).float().unsqueeze(0).unsqueeze(1)

    H_C = None
    generated = [start_char]

    for _ in range(length):
        outputs, H_C = model(x, H_C)
        logits = output_layer(outputs[-1]).squeeze(0)
        logits /= temperature

        probs = F.softmax(logits, dim=-1)
        idx = torch.multinomial(probs.view(-1), 1)

        generated.append(itos[idx.item()])
        x = F.one_hot(idx, len(stoi)).float().unsqueeze(0).unsqueeze(1)

    return "".join(generated)
