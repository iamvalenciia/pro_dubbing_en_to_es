"""
Diarization GPU Batching Optimization Patch
=============================================

This module provides a monkey-patch to improve GPU utilization during
speaker diarization by batching speaker-verification embedding calls.

Problem:
--------
ModelScope's SegmentationClusteringPipeline calls sv_pipeline([chunk], output_emb=True)
once per 1.5s audio window. For a 1-hour video, this results in ~4000 separate calls
with batch_size=1, causing GPU utilization to stay at 15-30% (kernel-launch bound).

Solution:
---------
Replace the pipeline's forward() method with a batching version that:
1. Extracts all audio segments upfront
2. Processes them in configurable batches (default: 16)
3. Collects embeddings efficiently
4. Proceeds with clustering

Expected Results:
-----------------
- Segment processing: ~4000 calls → ~250 calls (16x reduction)
- GPU utilization: 15-30% → 50-75%
- RTF improvement: 0.3-0.5x → 1.0-2.0x (2-4x faster)
- Accuracy: Unchanged (mathematically equivalent)

Usage:
------
    from videotrans.process.diarization_batching_patch import apply_diarization_batching
    
    ans = pipeline(task='speaker-diarization', model='...')
    apply_diarization_batching(ans, batch_size=16, num_speakers=num_speakers)
    result = ans(input_file)

Environment Variables:
----------------------
PYVIDEOTRANS_DIARIZATION_BATCH_SIZE: Override batch size (default: 16, 0=auto)
PYVIDEOTRANS_DIARIZATION_DEBUG: Enable debug logging (default: off)

Author: GPU Optimization Analysis
Version: 1.0
License: Same as pyvideotrans
"""

import os
import numpy as np
from videotrans.configure.config import logger


def apply_diarization_batching(pipeline, batch_size=16, num_speakers=-1):
    """
    Apply batching optimization to speaker-diarization pipeline.
    
    Args:
        pipeline: ModelScope SegmentationClusteringPipeline instance
        batch_size: Segments to batch per sv_pipeline call (default: 16, 0=auto)
        num_speakers: Oracle number of speakers for clustering (default: -1)
    
    Returns:
        Modified pipeline with optimized forward() method
    """
    if batch_size <= 0:
        batch_size = 16  # Auto batch size
    
    debug = os.environ.get('PYVIDEOTRANS_DIARIZATION_DEBUG', '0') == '1'
    
    # Store original forward
    _original_forward = pipeline.forward
    _stats = {'segments': 0, 'batches': 0, 'time_sv': 0}
    
    def _batched_forward(inputs):
        """Batched embedding extraction for precomputed diarization segments.

        IMPORTANT: `inputs` here is the list returned by ModelScope `chunk()` where each
        item is `[start_time_sec, end_time_sec, audio_chunk_np]`.
        We only batch SV embedding calls and keep the exact segment ordering/semantics.
        """
        import time
        
        try:
            if not isinstance(inputs, list) or len(inputs) == 0:
                return _original_forward(inputs)

            # Verify expected segment format [start, end, chunk]
            if not (isinstance(inputs[0], (list, tuple)) and len(inputs[0]) >= 3):
                return _original_forward(inputs)

            # Process segments in batches with SV pipeline
            t0 = time.time()
            embeddings = []
            stats_segments = len(inputs)
            stats_batches = 0

            for batch_idx in range(0, len(inputs), batch_size):
                batch_end = min(batch_idx + batch_size, len(inputs))
                batch_segments = inputs[batch_idx:batch_end]
                batch_chunks = [seg[2] for seg in batch_segments]
                stats_batches += 1
                
                if debug:
                    logger.debug(f'[DIAR-BATCH] Processing batch {stats_batches}: segments {batch_idx}-{batch_end - 1}')
                
                try:
                    batch_result = pipeline.sv_pipeline(batch_chunks, output_emb=True)

                    if isinstance(batch_result, dict) and 'embs' in batch_result:
                        embs = np.asarray(batch_result['embs'])
                        if embs.ndim == 1:
                            embs = embs.reshape(1, -1)

                        # Keep strict 1:1 mapping between segments and embeddings
                        if embs.shape[0] == len(batch_chunks):
                            embeddings.append(embs)
                        else:
                            raise RuntimeError(
                                f'Batch embeddings shape mismatch: got {embs.shape}, expected ({len(batch_chunks)}, d)'
                            )
                    else:
                        raise RuntimeError('SV batch result missing embs')

                except Exception as batch_err:
                    if debug:
                        logger.debug(f'[DIAR-BATCH] Batch processing failed, falling back to individual: {batch_err}')

                    # Fallback for this batch: preserve exact order
                    per_batch_embs = []
                    for seg in batch_segments:
                        try:
                            result = pipeline.sv_pipeline([seg[2]], output_emb=True)
                            emb = np.asarray(result['embs'])
                            if emb.ndim == 1:
                                emb = emb.reshape(1, -1)
                            if emb.shape[0] != 1:
                                raise RuntimeError(f'Expected (1,d) embedding, got {emb.shape}')
                            per_batch_embs.append(emb)
                        except Exception as e:
                            raise RuntimeError(f'SV extraction failed for one segment: {e}')

                    embeddings.append(np.concatenate(per_batch_embs, axis=0))

            t_sv = time.time() - t0
            _stats['segments'] = stats_segments
            _stats['batches'] = stats_batches
            _stats['time_sv'] = t_sv
            
            if debug:
                logger.debug(f'[DIAR-BATCH] SV extraction: {stats_batches} batches in {t_sv:.2f}s')

            # Concatenate embeddings and return to original pipeline flow.
            # Clustering + postprocess remain untouched in ModelScope.
            if embeddings:
                embeddings = np.concatenate(embeddings, axis=0)
            else:
                raise RuntimeError('No embeddings were produced in batched forward')

            if embeddings.shape[0] != len(inputs):
                raise RuntimeError(
                    f'Embedding count mismatch: got {embeddings.shape[0]}, expected {len(inputs)}'
                )

            reduction = _stats['segments'] / max(1, _stats['batches'])
            seg_dur = float(pipeline.config.get('seg_dur', 1.5))
            rtf = (_stats['segments'] * seg_dur) / max(0.1, _stats['time_sv'])
            
            logger.info(
                f'[DIAR-BATCH] Summary: segments={_stats["segments"]} → {_stats["batches"]} batches '
                f'({reduction:.1f}x reduction), RTF={rtf:.2f}x, sv={_stats["time_sv"]:.2f}s'
            )
            
            return embeddings
        
        except Exception as e:
            logger.exception(f'[DIAR-BATCH] Batched forward failed, falling back to original: {e}')
            return _original_forward(inputs)
    
    # Apply the patch
    pipeline.forward = _batched_forward
    logger.info(f'[DIAR-BATCH] Diarization batching optimization applied (batch_size={batch_size})')
    
    return pipeline
