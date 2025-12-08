import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel

def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device):
    """Decodes a sequence without teacher forcing."""
    
    if src_tokens.dtype == torch.bool:
        src_tokens, src_pad_mask = src_pad_mask, src_tokens
    if src_tokens.dtype != torch.long:
        src_tokens = src_tokens.long()
    
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    
    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        encoder_output = model.encoder(src_tokens, src_pad_mask)

    for t in range(max_out_len):
        max_len = model.decoder.pos_embed.size(1)
        if generated.size(1) > max_len:
            generated = generated[:, :max_len]
        
        trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)
        
        decoder_output = model.decoder(
            encoder_out=encoder_output, 
            src_mask=src_pad_mask, 
            trg=generated.long(), 
            trg_pad_mask=trg_pad_mask
        )
        output = decoder_output

        next_token_logits = output[:, -1, :]
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tokens], dim=1)
        finished = finished | (next_tokens.squeeze(1) == EOS)
        if finished.all():
            break
            
    predicted_tokens = []
    for seq in generated[:, 1:].tolist():
        if EOS in seq:
            idx = seq.index(EOS)
            seq = seq[:idx+1]
        predicted_tokens.append(seq)
    return predicted_tokens

def beam_search_decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
                       tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device, beam_size: int = 5, alpha: float = 0.7):
    """Beam Search with STOPPING CRITERION 2: Patience / N-Best."""
    
    if src_tokens.dtype == torch.bool:
        src_tokens, src_pad_mask = src_pad_mask, src_tokens
    if src_tokens.dtype != torch.long:
        src_tokens = src_tokens.long()
    
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()
    
    with torch.no_grad():
        encoder_output = model.encoder(src_tokens, src_pad_mask)

    beams = [(torch.tensor([[BOS]], device=device, dtype=torch.long), 0.0)]
    
    for _ in range(max_out_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue
            with torch.no_grad():
                max_len = model.decoder.pos_embed.size(1)
                if seq.size(1) > max_len:
                    seq = seq[:, :max_len]
                trg_pad_mask = (seq == PAD)[:, None, None, :]
                
                decoder_output = model.decoder(
                    encoder_out=encoder_output,
                    src_mask=src_pad_mask,
                    trg=seq.long(),
                    trg_pad_mask=trg_pad_mask
                )
                logits = decoder_output[:, -1, :]

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                new_seq = torch.cat([seq, topk_ids[:, k].unsqueeze(0)], dim=1)
                new_score = score + topk_log_probs[:, k].item()
                new_beams.append((new_seq, new_score))

        def get_score(seq, log_prob):
            if alpha == 0.0: return log_prob
            length = seq.size(1)
            lp = ((5 + length) ** alpha) / (6 ** alpha)
            return log_prob / lp

        beams = sorted(new_beams, key=lambda x: get_score(x[0], x[1]), reverse=True)[:beam_size]
        
        # === STOPPING CRITERION 2: Stop when at least N beams are finished ===
        # Let's say we wait for 50% of the beam to finish (e.g., 3 out of 5)
        finished_count = sum(1 for seq, _ in beams if seq[0, -1].item() == EOS)
        if finished_count >= 3:  # Hardcoded N=3 for beam_size=5
             break
        # ======================================================================
            
    best_seq, _ = beams[0]
    return [best_seq.squeeze(0).tolist()]
