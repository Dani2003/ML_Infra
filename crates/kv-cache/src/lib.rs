//! CPU-side model of a fixed-size paged KV block pool.

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AllocatorError {
    #[error("block pool capacity must be greater than zero")]
    ZeroCapacity,
    #[error("block pool capacity exceeds the physical block ID range")]
    CapacityTooLarge,
    #[error("host memory reservation failed for the block pool")]
    AllocationFailed,
    #[error("block size must be greater than zero")]
    ZeroBlockSize,
    #[error("no free KV blocks remain")]
    Exhausted,
    #[error("block ID {0} is outside the pool")]
    InvalidBlock(u32),
    #[error("block ID {0} is already free")]
    DoubleFree(u32),
}

#[derive(Debug, Default, Clone)]
pub struct BlockTable {
    physical_blocks: Vec<u32>,
    block_size: usize,
    token_count: usize,
}

impl BlockTable {
    pub fn new(block_size: usize) -> Result<Self, AllocatorError> {
        if block_size == 0 {
            return Err(AllocatorError::ZeroBlockSize);
        }
        Ok(Self { physical_blocks: Vec::new(), block_size, token_count: 0 })
    }

    pub fn physical_blocks(&self) -> &[u32] { &self.physical_blocks }
    pub fn token_count(&self) -> usize { self.token_count }
    pub fn block_size(&self) -> usize { self.block_size }

    pub fn physical_block_for_token(&self, token_index: usize) -> Option<(u32, usize)> {
        if token_index >= self.token_count { return None; }
        let logical = token_index / self.block_size;
        let offset = token_index % self.block_size;
        self.physical_blocks.get(logical).copied().map(|physical| (physical, offset))
    }

    pub fn push_block(&mut self, block_id: u32) {
        self.physical_blocks.push(block_id);
    }

    pub fn append_token(&mut self) {
        self.token_count += 1;
    }

    pub fn needs_new_block(&self) -> bool {
        self.token_count % self.block_size == 0
    }

    pub fn drain_blocks(&mut self) -> Vec<u32> {
        self.token_count = 0;
        self.physical_blocks.drain(..).collect()
    }
}

#[derive(Debug)]
pub struct BlockPool {
    allocated: Vec<bool>,
    free: Vec<u32>,
}

impl BlockPool {
    pub fn new(num_blocks: usize) -> Result<Self, AllocatorError> {
        if num_blocks == 0 { return Err(AllocatorError::ZeroCapacity); }
        if num_blocks > u32::MAX as usize { return Err(AllocatorError::CapacityTooLarge); }
        let mut free = Vec::new();
        free.try_reserve_exact(num_blocks).map_err(|_| AllocatorError::AllocationFailed)?;
        for id in (0..num_blocks as u32).rev() { free.push(id); }
        let mut allocated = Vec::new();
        allocated.try_reserve_exact(num_blocks).map_err(|_| AllocatorError::AllocationFailed)?;
        allocated.resize(num_blocks, false);
        Ok(Self { allocated, free })
    }

    pub fn capacity(&self) -> usize { self.allocated.len() }
    pub fn free_count(&self) -> usize { self.free.len() }

    pub fn allocate(&mut self) -> Result<u32, AllocatorError> {
        let id = self.free.pop().ok_or(AllocatorError::Exhausted)?;
        self.allocated[id as usize] = true;
        Ok(id)
    }

    pub fn release(&mut self, id: u32) -> Result<(), AllocatorError> {
        let Some(is_allocated) = self.allocated.get_mut(id as usize) else {
            return Err(AllocatorError::InvalidBlock(id));
        };
        if !*is_allocated { return Err(AllocatorError::DoubleFree(id)); }
        *is_allocated = false;
        self.free.push(id);
        Ok(())
    }

    pub fn append_token_block(&mut self, table: &mut BlockTable) -> Result<(), AllocatorError> {
        if table.block_size == 0 { return Err(AllocatorError::ZeroBlockSize); }
        let next_token_count = table.token_count.checked_add(1).ok_or(AllocatorError::Exhausted)?;
        if table.token_count % table.block_size == 0 {
            table.physical_blocks.push(self.allocate()?);
        }
        table.token_count = next_token_count;
        Ok(())
    }

    pub fn release_table(&mut self, table: &mut BlockTable) -> Result<(), AllocatorError> {
        // Validate every ID first, so a bad table cannot cause a partial release.
        for &id in &table.physical_blocks {
            match self.allocated.get(id as usize) {
                None => return Err(AllocatorError::InvalidBlock(id)),
                Some(false) => return Err(AllocatorError::DoubleFree(id)),
                Some(true) => {}
            }
        }
        for id in table.physical_blocks.drain(..) {
            self.allocated[id as usize] = false;
            self.free.push(id);
        }
        table.token_count = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_capacity_rejected() {
        assert_eq!(BlockPool::new(0).unwrap_err(), AllocatorError::ZeroCapacity);
    }

    #[test]
    fn test_zero_block_size_rejected() {
        assert_eq!(BlockTable::new(0).unwrap_err(), AllocatorError::ZeroBlockSize);
    }

    #[test]
    fn test_block_allocation_and_release() {
        let mut pool = BlockPool::new(3).unwrap();
        assert_eq!(pool.capacity(), 3);
        assert_eq!(pool.free_count(), 3);

        let b0 = pool.allocate().unwrap();
        let b1 = pool.allocate().unwrap();
        let b2 = pool.allocate().unwrap();

        assert_eq!(pool.free_count(), 0);
        assert_eq!(pool.allocate().unwrap_err(), AllocatorError::Exhausted);

        // Double free check
        pool.release(b0).unwrap();
        assert_eq!(pool.release(b0).unwrap_err(), AllocatorError::DoubleFree(b0));

        // Invalid block check
        assert_eq!(pool.release(999).unwrap_err(), AllocatorError::InvalidBlock(999));

        pool.release(b1).unwrap();
        pool.release(b2).unwrap();
        assert_eq!(pool.free_count(), 3);
    }

    #[test]
    fn test_block_table_token_mapping() {
        let mut pool = BlockPool::new(10).unwrap();
        let block_size = 4;
        let mut table = BlockTable::new(block_size).unwrap();

        // Append 10 tokens across block boundaries (ceil(10/4) = 3 physical blocks)
        for i in 0..10 {
            pool.append_token_block(&mut table).unwrap();
            let (phys, offset) = table.physical_block_for_token(i).expect("token must map");
            let logical_block = i / block_size;
            let expected_offset = i % block_size;
            assert_eq!(offset, expected_offset);
            assert_eq!(phys, table.physical_blocks()[logical_block]);
        }

        assert_eq!(table.physical_blocks().len(), 3);
        assert_eq!(table.token_count(), 10);
        assert!(table.physical_block_for_token(10).is_none());

        // Release entire table
        pool.release_table(&mut table).unwrap();
        assert_eq!(pool.free_count(), 10);
        assert_eq!(table.physical_blocks().len(), 0);
        assert_eq!(table.token_count(), 0);
    }
}

