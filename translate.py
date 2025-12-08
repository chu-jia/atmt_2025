import os
import logging
import argparse
import time
import numpy as np
import sacrebleu
from tqdm import tqdm
import torch
import sentencepiece as spm
from torch.serialization import default_restore_location
import sys

# Ensure seq2seq can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from seq2seq.decode import beam_search_decode, decode
from seq2seq.data.tokenizer import BPETokenizer
from seq2seq import models, utils

def get_args():
    """ Defines generation-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', action='store_true', help='Use a GPU')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')
    
    # Add data arguments
    parser.add_argument('--input', required=True, help='Path to the raw text file to translate (one sentence per line)')
    parser.add_argument('--src-tokenizer', help='path to source sentencepiece tokenizer', required=True)
    parser.add_argument('--tgt-tokenizer', help='path to target sentencepiece tokenizer', required=True)
    parser.add_argument('--checkpoint-path', required=True, help='path to the model file')
    parser.add_argument('--batch-size', default=1, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--output', required=True, type=str, help='path to the output file destination')
    parser.add_argument('--max-len', default=128, type=int, help='maximum length of generated sequence')
    
    # Beam search decoding parameters
    parser.add_argument('--beam-size', default=1, type=int, help='beam size for beam search decoding')
    parser.add_argument('--alpha', default=0.7, type=float, help='length normalization hyperparameter for beam search')
    
    # BLEU computation arguments
    parser.add_argument('--bleu', action='store_true', help='If set, compute BLEU score after translation')
    parser.add_argument('--reference', type=str, help='Path to the reference file (one sentence per line, required if --bleu is set)')
    
    return parser.parse_args()

def postprocess_ids(ids, pad, bos, eos):
    """Remove leading BOS, truncate at first EOS, remove PADs."""
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if len(ids) > 0 and ids[0] == bos:
        ids = ids[1:]
    if eos in ids:
        ids = ids[:ids.index(eos)]
    ids = [i for i in ids if i != pad]
    return ids

def decode_sentence(tokenizer, sentence_ids):
    PAD = tokenizer.pad_id()
    BOS = tokenizer.bos_id()
    EOS = tokenizer.eos_id()
    ids = postprocess_ids(sentence_ids, PAD, BOS, EOS)
    return tokenizer.Decode(ids)

def main(args):
    """ Main translation function """
    torch.manual_seed(args.seed)
    
    # Load checkpoint safely
    if args.cuda and torch.cuda.is_available():
        map_location = None
    else:
        map_location = 'cpu'
        
    logging.info(f"Loading checkpoint from {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location=map_location)
    
    # Merge args
    if 'args' in state_dict:
        model_args = state_dict['args']
        for k, v in vars(model_args).items():
            if k not in vars(args):
                setattr(args, k, v)
    
    utils.init_logging(args)
    DEVICE = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Load tokenizers
    try:
        src_tokenizer = spm.SentencePieceProcessor(model_file=args.src_tokenizer)
        tgt_tokenizer = spm.SentencePieceProcessor(model_file=args.tgt_tokenizer)
    except Exception:
        from seq2seq.data.tokenizer import load_tokenizer
        src_tokenizer = load_tokenizer(args.src_tokenizer)
        tgt_tokenizer = load_tokenizer(args.tgt_tokenizer)

    # Build model
    model = models.build_model(args, src_tokenizer, tgt_tokenizer)
    model.to(DEVICE)
    model.eval()
    
    if 'model' in state_dict:
        model.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict)

    # Read input sentences
    with open(args.input, encoding="utf-8") as f:
        src_lines = [line.strip() for line in f if line.strip()]

    # === FIX 1: Correctly Encode Source (Add EOS!) ===
    # Using EncodeAsIds usually misses EOS. We manually add it if needed.
    src_encoded = []
    for line in src_lines:
        ids = src_tokenizer.EncodeAsIds(line)
        # Ensure EOS is at the end (Training usually relies on this)
        if len(ids) == 0 or ids[-1] != src_tokenizer.eos_id():
            ids.append(src_tokenizer.eos_id())
        src_encoded.append(torch.tensor(ids, dtype=torch.long))
    # =================================================

    PAD = src_tokenizer.pad_id()
    print(f'PAD ID: {PAD}')

    if args.output is not None:
        with open(args.output, 'w', encoding="utf-8") as out_file:
            out_file.write('')

    translations = []
    start_time = time.perf_counter()

    def batch_iter(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i+batch_size]

    for batch in tqdm(batch_iter(src_encoded, args.batch_size)):
        with torch.no_grad():
            batch_lengths = [len(x) for x in batch]
            max_len = max(batch_lengths)
            batch_padded = [
                torch.cat([x, torch.full((max_len - len(x),), PAD, dtype=torch.long)]) if len(x) < max_len else x
                for x in batch
            ]
            src_tokens = torch.stack(batch_padded).to(DEVICE)

            # === FIX 2: Correct Mask Logic ===
            # Create mask: (batch, 1, 1, seq_len)
            # PyTorch Transformer typically uses: True for Padding (Ignore)
            src_pad_mask = (src_tokens == PAD).unsqueeze(1).unsqueeze(2)
            trg_pad_mask = None
            # =================================

            if args.beam_size == 1:
                prediction = decode(model=model,
                                    src_tokens=src_tokens,
                                    src_pad_mask=src_pad_mask,
                                    max_out_len=args.max_len,
                                    tgt_tokenizer=tgt_tokenizer,
                                    args=args,
                                    device=DEVICE)
            else:
                prediction = beam_search_decode(model=model,
                                                src_tokens=src_tokens,
                                                src_pad_mask=src_pad_mask,
                                                max_out_len=args.max_len,
                                                tgt_tokenizer=tgt_tokenizer,
                                                args=args,
                                                device=DEVICE,
                                                beam_size=args.beam_size,
                                                alpha=args.alpha)

            for sent in prediction:
                if isinstance(sent, torch.Tensor):
                    sent = sent.tolist()
                translation = decode_sentence(tgt_tokenizer, sent)
                translations.append(translation)
                
                if args.output is not None:
                    with open(args.output, 'a', encoding="utf-8") as out_file:
                        out_file.write(translation + '\n')

    logging.info(f'Wrote {len(translations)} lines to {args.output}')
    end_time = time.perf_counter()
    logging.info(f'Translation completed in {end_time - start_time:.2f} seconds')

    if getattr(args, 'bleu', False):
        if not args.reference:
             print("Warning: --bleu set but no --reference provided.")
        else:
            with open(args.reference, encoding='utf-8') as ref_file:
                references = [line.strip() for line in ref_file if line.strip()]
            bleu = sacrebleu.corpus_bleu(translations, [references])
            print(f"BLEU score: {bleu.score:.2f}")

if __name__ == '__main__':
    args = get_args()
    if getattr(args, 'bleu', False) and not args.reference:
         raise ValueError("You must provide --reference when using --bleu.")
    main(args)
