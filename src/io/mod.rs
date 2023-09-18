use parking_lot::RwLock;
use std::sync::{Arc, LazyLock};
use vm_memory::{
    GuestAddress, GuestAddressSpace, GuestMemoryAtomic, GuestMemoryMmap, GuestRegionMmap,
    MmapRegion,
};

pub static BUS: LazyLock<RwLock<Bus>> = LazyLock::new(|| RwLock::new(Bus::new(0)));

#[derive(Debug, Clone, Copy, bytemuck::NoUninit, PartialEq, Eq)]
#[repr(C)]
pub struct Region(pub u64, pub usize);

#[derive(Clone)]
pub struct Dram {
    pub size: u64,
    pub memory: GuestMemoryAtomic<GuestMemoryMmap>,
}

impl Dram {
    pub fn new(size: u64) -> Self {
        let arc = Arc::new(GuestMemoryMmap::new());
        Self {
            size,
            memory: arc.clone().into(),
        }
    }

    pub fn add_region(&self, region: &Region) {
        if (region.0 + region.1 as u64) > self.size {
            panic!("out of bounds memory access");
        }

        let mmap = Arc::new(
            GuestRegionMmap::new(MmapRegion::new(region.1).unwrap(), GuestAddress(region.0))
                .unwrap(),
        );
        let memory = self.memory.memory().insert_region(mmap).unwrap();
        self.memory.lock().unwrap().replace(memory);
    }

    pub fn remove_region(&self, region: &Region) {
        if (region.0 + region.1 as u64) > self.size {
            panic!("out of bounds memory access");
        }

        let memory = self
            .memory
            .memory()
            .remove_region(GuestAddress(region.0), region.1 as u64)
            .unwrap()
            .0;
        self.memory.lock().unwrap().replace(memory);
    }
}

#[derive(Clone)]
pub struct Bus {
    pub dram: Dram,
}

impl Bus {
    pub fn new(dram_size: u64) -> Self {
        Self {
            dram: Dram::new(dram_size),
        }
    }
}
