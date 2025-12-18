def train(text, epochs=20, seq_len=40, hidden_size=128, lr=0.003):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stoi, itos, vocab_size = build_vocab(text)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    # ---- train / validation split ----
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    model = LSTMScratch(vocab_size, hidden_size).to(device)
    output_layer = nn.Linear(hidden_size, vocab_size).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(output_layer.parameters()), lr=lr
    )

    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        H_C = None

        for step, i in enumerate(range(0, len(train_data) - seq_len - 1, seq_len)):
            x_idx = train_data[i:i+seq_len]
            y = train_data[i+1:i+seq_len+1]

            x = one_hot(x_idx, vocab_size).to(device).unsqueeze(1)
            y = y.to(device)

            outputs, H_C = model(x, H_C)
            H_C = (H_C[0].detach(), H_C[1].detach())

            logits = torch.stack([output_layer(h) for h in outputs]).squeeze(1)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += y.numel()

            if step % 500 == 0:
                print(
                    f"Epoch {epoch:02d} | Step {step:06d} | "
                    f"Batch ppl: {math.exp(loss.item()/y.numel()):.2f}",
                    flush=True
                )

        train_ppl = math.exp(total_loss / total_tokens)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        H_C = None

        with torch.no_grad():
            for i in range(0, len(val_data) - seq_len - 1, seq_len):
                x_idx = val_data[i:i+seq_len]
                y = val_data[i+1:i+seq_len+1]

                x = one_hot(x_idx, vocab_size).to(device).unsqueeze(1)
                y = y.to(device)

                outputs, H_C = model(x, H_C)
                logits = torch.stack([output_layer(h) for h in outputs]).squeeze(1)

                loss = loss_fn(logits, y)
                val_loss += loss.item()
                val_tokens += y.numel()

        val_ppl = math.exp(val_loss / val_tokens)

        print(f"\nEpoch {epoch:02d} DONE")
        print(f"Train Perplexity: {train_ppl:.2f}")
        print(f"Valid Perplexity: {val_ppl:.2f}")

        sample = generate_text(
            model, output_layer, stoi, itos,
            start_char=text[0],
            device=device
        )
        print("\n--- Sample ---")
        print(sample)
        print("-" * 60)