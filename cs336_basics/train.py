import torch
from .bpe_tokenizer import BPETokenizer
from .models.transformer import Transformer
from .models.optimizers import AdamW
from .utils import *
import argparse
import wandb
from .train_bpe import load_tokenizer
from .common_config import DATA_PATH, RESULT_PATH
from time import time
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--residual_pdrop", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="tinystories")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--total_tokens", type=int, default=12800000)
    parser.add_argument("--post_norm", action="store_true", dest="pre_norm",
                        help="Post-normalization")
    parser.add_argument("--layer_norm", action="store_false")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--group_name", type=str, default="hyperparameter_search")
    parser.add_argument("--disable_wandb", action="store_true")

    args = parser.parse_args()

    print(args)
    torch.set_float32_matmul_precision('high')

    args.num_steps = args.total_tokens // args.batch_size

    run_name = f'lr{args.lr}_bs{args.batch_size}'
    train_data_path = f"{DATA_PATH}/{args.dataset}_train.npy"
    valid_data_path = f"{DATA_PATH}/{args.dataset}_valid.npy"
    checkpoint_path = f"{DATA_PATH}/ckps/{args.checkpoint}"
    if args.checkpoint == "":
        checkpoint_path = f"{DATA_PATH}/ckps/{run_name}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
 
    # Load the data 
    train_data, valid_data = np.load(train_data_path, mmap_mode='r'), np.load(valid_data_path, mmap_mode='r')

    # Load the tokenizer
    tokenizer = load_tokenizer(vocab_path=f"{RESULT_PATH}/{args.dataset}_vocab.json", merges_path=f"{RESULT_PATH}/{args.dataset}_merges.txt", special_tokens=[])
    vocab_size = len(tokenizer.vocab)
    print(f"Vocab Size: {vocab_size}")

    # Load the model
    model = Transformer(
        vocab_size=vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
    )
    model.to(device)
    
    # compile the model
    model = torch.compile(model)

    # Load the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Initialize the wandb
    if not args.disable_wandb:
        wandb.init(project="cs336_a1", config=args, name=run_name, group=args.group_name)

    # Load the loss function
    loss_fn = cross_entropy_loss

    current_step = 0
    # Load the checkpoint
    if args.resume:
        current_step = load_checkpoint(args.checkpoint, model, optimizer)

    def validate():
        """Validate the model on the validation set."""
        # Load the data
        model.eval()
        with torch.no_grad():
            source, target = load_data(valid_data, args.val_batch_size, args.context_length, device)
            
            # Load the loss
            logits = model(source)
            loss = cross_entropy_loss(logits, target)
            perpl = perplexity(logits, target)
            
            if not args.disable_wandb:
                wandb.log({"valid_loss": loss, "valid_perplexity": perpl})
            
            model.train()	
            return loss.item(), perpl
    
    start_time = time()
    
    # Train the model
    while current_step < args.num_steps:
        source, target = load_data(train_data, args.batch_size, args.context_length, device)
        logits = model(source)
        loss = loss_fn(logits, target)
        
        optimizer.zero_grad()
        loss.backward()
        gradient_clip(model.parameters(), args.grad_clip)
        optimizer.step()
        
        if (current_step + 1) % 100 == 0:
            print(f"Step: {current_step}, Loss: {loss.item()}")
   
        if not args.disable_wandb:
            wandb.log({"loss": loss.item(), "time": time() - start_time, "step": current_step})
    
        if (current_step + 1) % 500 == 0:
            # Evaluate the model
            valid_loss, perpl = validate()
            print(f"Validation Loss: {valid_loss}, Perplexity: {perpl}")
        
        if (current_step + 1) % 10000 == 0:
            save_checkpoint(model, optimizer, current_step, checkpoint_path)
        
        current_step += 1
    
    # Save the final model
    save_checkpoint(model, optimizer, current_step, checkpoint_path)
    
if __name__ == "__main__":
    main()
    # tokenizer = load_tokenizer(vocab_path=f"{RESULT_PATH}/{args.dataset}_vocab.json", merges_path=f"{RESULT_PATH}/{args.dataset}_merges.txt", special_tokens=[])
    # checkpoint_path = f"{DATA_PATH}/ckps/"
    # model = Transformer(vocab_size=100, context_length=256, num_layers=4, d_model=512, num_heads=16, d_ff=2048, attn_pdrop=0.1, residual_pdrop=0.1, parallel=False, layer_norm=True, pre_norm=True)
    # optimizer = AdamW(model.parameters(), lr=1e-3)
    # load_checkpoint(checkpoint_path, model, optimizer)
    # text = "The quick brown fox jumps over the lazy dog"
    # ids = tokenizer.encode(text) # (seq_len,)
    # generated_ids = model.generate(ids, max_len=100, temperature=0.7, top_p=0.9)
    # print(tokenizer.decode(generated_ids))	