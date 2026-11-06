//! Execution contract shared by the scheduler and model backends.

pub mod ffi;

use thiserror::Error;

#[derive(Clone, Debug)]
pub struct DecodeInput {
    pub request_id: u64,
    pub prompt: String,
    pub generated_tokens: usize,
    pub block_table: Vec<i32>,
    pub context_len: usize,
}

#[derive(Clone, Debug)]
pub struct DecodeOutput {
    /// Text emitted for this request at this decode iteration.
    pub token: String,
    pub finished: bool,
}

#[derive(Debug, Error)]
#[error("model runner failed: {0}")]
pub struct RunnerError(pub String);

/// A synchronous step boundary keeps the scheduler contract small. Real GPU
/// runners should implement this with an execution thread or a nonblocking
/// backend so device waits do not stall Tokio worker threads.
pub trait ModelRunner: Send + Sync + 'static {
    fn decode_step(&self, batch: &[DecodeInput]) -> Result<Vec<DecodeOutput>, RunnerError>;
}

/// Deterministic stand-in used to exercise the request and batching path.
#[derive(Default)]
pub struct MockRunner;

impl ModelRunner for MockRunner {
    fn decode_step(&self, batch: &[DecodeInput]) -> Result<Vec<DecodeOutput>, RunnerError> {
        Ok(batch
            .iter()
            .map(|request| DecodeOutput {
                token: format!(" t{}", request.generated_tokens + 1),
                finished: false,
            })
            .collect())
    }
}

use std::sync::Arc;
use crate::ffi::GpuKVCacheManager;

/// PagedAttention hardware runner that coordinates with GpuKVCacheManager
pub struct PagedAttentionRunner {
    pub cache_manager: Arc<GpuKVCacheManager>,
}

impl PagedAttentionRunner {
    pub fn new(cache_manager: Arc<GpuKVCacheManager>) -> Self {
        Self { cache_manager }
    }
}

impl ModelRunner for PagedAttentionRunner {
    fn decode_step(&self, batch: &[DecodeInput]) -> Result<Vec<DecodeOutput>, RunnerError> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        let num_seqs = batch.len();
        let num_heads = self.cache_manager.num_heads();
        let head_dim = self.cache_manager.head_dim();

        let max_blocks = batch
            .iter()
            .map(|req| req.block_table.len())
            .max()
            .unwrap_or(1)
            .max(1);

        let mut block_tables = vec![0i32; num_seqs * max_blocks];
        let mut context_lens = Vec::with_capacity(num_seqs);

        for (i, req) in batch.iter().enumerate() {
            context_lens.push(req.context_len as i32);
            for (j, &block_id) in req.block_table.iter().enumerate() {
                block_tables[i * max_blocks + j] = block_id;
            }
        }

        let query = vec![1.0f32; num_seqs * num_heads * head_dim];
        let mut output = vec![0.0f32; num_seqs * num_heads * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();

        self.cache_manager.decode_attention(
            &query,
            &block_tables,
            &context_lens,
            max_blocks,
            scale,
            &mut output,
        );

        Ok(batch
            .iter()
            .map(|req| DecodeOutput {
                token: format!(" t{}", req.generated_tokens + 1),
                finished: false,
            })
            .collect())
    }
}
