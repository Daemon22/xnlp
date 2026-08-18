#!/usr/bin/env python
"""Run the XNLP evaluation suite on a checkpoint."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True)

from xnlp_trainer.evaluate import XNLPEvaluation

ckpt = sys.argv[1] if len(sys.argv) > 1 else 'outputs/best_model.pt'
evaluator = XNLPEvaluation(ckpt)
evaluator.run_all()
