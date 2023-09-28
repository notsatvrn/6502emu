use crate::instructions::riscv32::*;
use crate::io::*;

use parking_lot::{Mutex, RwLock};
use std::sync::LazyLock;
use std::{collections::BTreeSet, sync::Arc};
use tinyvec::ArrayVec;
use vm_memory::{Bytes, GuestAddress, GuestAddressSpace, GuestMemory, GuestMemoryRegion};

#[cfg(feature = "debug")]
use std::mem::transmute;

pub static CPU: LazyLock<RwLock<Cpu>> = LazyLock::new(|| RwLock::new(Cpu::default()));

#[derive(Default, Debug)]
pub struct Cpu {
    pub cores: Vec<Arc<Mutex<Core>>>,
}

pub fn init(core_count: usize, hart_count: usize, memory_size: u32) {
    let mut cpu = CPU.write();

    if !cpu.cores.is_empty() {
        panic!("init called after already initialized");
    }

    BUS.write().dram.size = memory_size as u64;

    let mut csrs = [0u32; 4096];

    // machine info registers
    // mvendorid (0xF11) is already 0, keep because this is a non-commercial implementation
    csrs[0xF12] = 35; // marchid

    // machine trap setup
    csrs[0x301] = 0b01000000000000000001000100101100; // misa

    let mut core_i: usize = 0;
    let mut hart_i: usize = 0;
    while core_i < core_count {
        let mut harts = HartVec::new();

        while hart_i < hart_count * (core_i + 1) {
            // hart-specific csrs
            csrs[0xF14] = hart_i as u32; // mhartid

            harts.push(Arc::new(Mutex::new(Hart {
                region: Region(0, 0),
                x: [0; 32],
                f: [0.0; 32],
                csrs,
                pc: 0,
                pipeline: ArrayVec::new(),
            })));

            hart_i += 1;
        }

        cpu.cores.push(Arc::new(Mutex::new(Core {
            harts,
            running: BTreeSet::new(),
        })));

        core_i += 1;
    }
}

// replace this whole thing with a proper MMU
pub fn prepare_hart(core_id: usize, hart_id: usize, data: Vec<u8>) {
    let bus = BUS.read();

    // step 1: get address
    let mut address = 0;
    let memory = bus.dram.memory.memory();
    let data_len = data.len() as u64;
    let mut last_end = 0;

    for (i, region) in memory.iter().enumerate() {
        let start = region.start_addr().0;

        if i == 0 && data_len < start {
            break;
        }

        if (start - last_end) >= data_len {
            address = last_end;
            break;
        }

        last_end = start + region.len();

        if (i + 1) == memory.num_regions() {
            if (last_end + data_len) < bus.dram.size {
                address = last_end;
            } else {
                panic!("out of bounds memory access");
            }
        }
    }

    // step 2: load data
    let mut cpu = CPU.write();
    let mut core = cpu.cores.get_mut(core_id).unwrap().lock();
    let mut hart = core.harts.get_mut(hart_id).unwrap().lock();
    hart.region = Region(address, data_len as usize);

    bus.dram.add_region(&hart.region);
    drop(hart);

    bus.dram
        .memory
        .memory()
        .write(data.as_slice(), GuestAddress(address))
        .unwrap();

    // step 3: declare hart as running
    core.running.insert(hart_id);
}

type HartVec = ArrayVec<[Arc<Mutex<Hart>>; 4]>;

#[derive(Debug)]
pub struct Core {
    pub harts: HartVec,
    pub running: BTreeSet<usize>,
}

impl Core {
    pub fn stop_hart(&mut self, hart: usize) {
        if self.running.contains(&hart) {
            self.harts[hart].lock().stop();
            self.running.remove(&hart);
        }
    }
}

/*
#[derive(Debug, PartialEq, Eq)]
pub enum Register {
    FloatingPoint(usize),
    Integer(usize),
}
*/

#[derive(Debug, PartialEq, Eq)]
pub enum Stage {
    Fetch,
    Decode,
    Execute,
    Memory,
    Write,
}

#[derive(Debug, PartialEq)]
pub enum ReturnMode {
    Integer(usize, u32),
    Single(usize, f32),
    Double(usize, f64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryOperation {
    Load(u32, LoadMode),
    Store(u32, ArrayVec<[u8; 8]>),
}

#[derive(Debug)]
pub struct InstructionData {
    pub insti: u32,
    pub inst: Instruction,
    pub stage: Stage,
    pub rd: Option<ReturnMode>,
    pub memop: Option<MemoryOperation>,
}

impl Default for InstructionData {
    fn default() -> Self {
        Self {
            insti: 0,
            inst: Instruction::Compressed(CompressedInstruction::NOP),
            stage: Stage::Fetch,
            rd: None,
            memop: None,
        }
    }
}

#[derive(Debug)]
pub struct Hart {
    pub region: Region,
    pub x: [u32; 32],
    pub f: [f64; 32],
    pub csrs: [u32; 4096],
    pub pc: u32,
    pub pipeline: ArrayVec<[InstructionData; 5]>,
}

impl Default for Hart {
    #[inline]
    fn default() -> Self {
        Self {
            region: Region::default(),
            x: [0; 32],
            f: [0.0; 32],
            csrs: [0; 4096],
            pc: 0,
            pipeline: ArrayVec::default(),
        }
    }
}

impl Hart {
    pub fn stop(&mut self) {
        BUS.read().dram.remove_region(&self.region);
        self.region = Region(0, 0);
        self.x = [0; 32];
        self.f = [0.0; 32];
        self.csrs = [0; 4096];
        self.pc = 0;
    }

    pub fn fetch(&mut self) {
        let index = self.region.0 + self.pc as u64;
        #[cfg(feature = "debug")]
        println!(
            "start {} | pc: {} | index: {}",
            self.region.0, self.pc, index
        );

        let bus = BUS.read();
        let mem = bus.dram.memory.memory();
        let mut buf = [0u8; 4];
        let res = mem.read(&mut buf, GuestAddress(index)).unwrap();
        drop(bus);

        let insti = if buf[0] & 0b11 == 0b11 {
            if res != 4 {
                panic!("reached end of program early");
            }

            self.pc += 4;
            u32::from_le_bytes(buf)
        } else {
            if res != 2 {
                panic!("reached end of program early");
            }

            self.pc += 2;
            u16::from_le_bytes([buf[0], buf[1]]) as u32
        };

        self.pipeline.insert(
            0,
            InstructionData {
                insti,
                stage: Stage::Decode,
                ..Default::default()
            },
        );
    }

    pub fn decode(&mut self) {
        let inst_data = self.pipeline.get_mut(1).unwrap();
        inst_data.inst = if inst_data.insti & 0b11 == 0b11 {
            Instruction::Full(FullInstruction::decode(inst_data.insti))
        } else {
            Instruction::Compressed(CompressedInstruction::decode(inst_data.insti as u16))
        };
        inst_data.stage = Stage::Execute;
    }

    // Execute an instruction.

    #[inline]
    pub fn execute(&mut self) {
        match self.pipeline[2].inst {
            Instruction::Full(full) => self.execute_full(full),
            Instruction::Compressed(compressed) => self.execute_compressed(compressed),
        };

        self.pipeline.get_mut(2).unwrap().stage = Stage::Memory;
    }

    // Execute a compressed instruction.

    pub fn execute_compressed(&mut self, inst: CompressedInstruction) {
        #[cfg(feature = "debug")]
        println!("{}", inst);

        match inst {
            CompressedInstruction::NOP => {}
            _ => unimplemented!(),
        }
    }

    // Execute a full instruction.

    pub fn execute_full(&mut self, inst: FullInstruction) {
        #[cfg(feature = "debug")]
        println!("{}", inst);

        match inst {
            FullInstruction::LUI(rd, immediate) => {
                self.set(rd, immediate - 4);
            }
            FullInstruction::AUIPC(rd, immediate) => {
                self.pc += immediate;
                self.set(rd, self.pc);
            }
            FullInstruction::JAL(rd, offset) => {
                self.pc = self.pc.wrapping_add_signed(offset);
                self.set(rd, self.pc + 4);
            }
            FullInstruction::JALR(rd, offset, rs1) => {
                self.pc = self.get(rs1).wrapping_add_signed(offset) & 0xFFFFFFFE;
                self.set(rd, self.pc + 4);
            }
            FullInstruction::Branch(Branch {
                rs1,
                rs2,
                offset,
                mode,
            }) => {
                let branching = match mode {
                    BranchMode::Equal => self.get(rs1) == self.get(rs2),
                    BranchMode::NotEqual => self.get(rs1) != self.get(rs2),
                    BranchMode::LessThan => (self.get(rs1) as i32) < self.get(rs2) as i32,
                    BranchMode::GreaterOrEqual => (self.get(rs1) as i32) >= self.get(rs2) as i32,
                    BranchMode::LessThanUnsigned => self.get(rs1) < self.get(rs2),
                    BranchMode::GreaterOrEqualUnsigned => self.get(rs1) >= self.get(rs2),
                };

                if branching {
                    self.pc = self.pc.wrapping_add_signed(offset);
                    #[cfg(feature = "debug")]
                    println!(" | branched");
                } else {
                    #[cfg(feature = "debug")]
                    println!(" | didn't branch");
                }
            }
            FullInstruction::Load(Load {
                rd,
                rs1,
                offset,
                mode,
            }) => {
                let address = self.get(rs1).wrapping_add_signed(offset) + self.region.0 as u32;
                let ret = match mode {
                    LoadMode::Single => ReturnMode::Single(rd, 0.0),
                    LoadMode::Double => ReturnMode::Double(rd, 0.0),
                    _ => ReturnMode::Integer(rd, 0),
                };
                self.pipeline[2].rd = Some(ret);
                self.pipeline[2].memop = Some(MemoryOperation::Load(address, mode));
            }
            FullInstruction::Store(Store {
                rs1,
                rs2,
                offset,
                mode,
            }) => {
                let bytes: ArrayVec<[u8; 8]> = match mode {
                    StoreMode::Byte => [self.get(rs2) as u8].as_ref().try_into().unwrap(),
                    StoreMode::HalfWord => (self.get(rs2) as u16)
                        .to_le_bytes()
                        .as_ref()
                        .try_into()
                        .unwrap(),
                    StoreMode::Word => self.get(rs2).to_le_bytes().as_ref().try_into().unwrap(),
                    StoreMode::Single => {
                        self.get_f32(rs2).to_le_bytes().as_ref().try_into().unwrap()
                    }
                    StoreMode::Double => {
                        self.get_f64(rs2).to_le_bytes().as_ref().try_into().unwrap()
                    }
                };

                let address = self.get(rs1).wrapping_add_signed(offset) + self.region.0 as u32;
                self.pipeline[2].memop = Some(MemoryOperation::Store(address, bytes));
            }
            FullInstruction::IMMOp(IMMOp {
                rd,
                rs1,
                immediate,
                mode,
            }) => {
                if rd == 0 {
                    return;
                }

                self.set(
                    rd,
                    match mode {
                        IMMOpMode::Add => (self.get(rs1) as i32).wrapping_add(immediate) as u32,
                        IMMOpMode::SetLessThan => ((self.get(rs1) as i32) < immediate) as u32,
                        IMMOpMode::SetLessThanUnsigned => (self.get(rs1) < immediate as u32) as u32,
                        IMMOpMode::ExclusiveOr => self.get(rs1) ^ (immediate as u32),
                        IMMOpMode::Or => self.get(rs1) | (immediate as u32),
                        IMMOpMode::And => self.get(rs1) & (immediate as u32),
                    },
                );
            }
            FullInstruction::IMMShift(IMMShift {
                rd,
                rs1,
                amount,
                mode,
                ..
            }) => {
                if rd == 0 {
                    return;
                }

                self.set(
                    rd,
                    match mode {
                        ShiftMode::LogicalLeft => self.get(rs1) << amount,
                        ShiftMode::LogicalRight => self.get(rs1) >> amount,
                        ShiftMode::ArithmeticRight => ((self.get(rs1) as i32) >> amount) as u32,
                    },
                );
            }
            FullInstruction::IntOp(IntOp { rd, rs1, rs2, mode }) => {
                if rd == 0 {
                    return;
                }

                self.set(
                    rd,
                    match mode {
                        IntOpMode::Add => self.get(rs1).wrapping_add(self.get(rs2)),
                        IntOpMode::Subtract => self.get(rs1).wrapping_sub(self.get(rs2)),
                        IntOpMode::SetLessThan => {
                            ((self.get(rs1) as i32) < (self.get(rs2) as i32)) as u32
                        }
                        IntOpMode::SetLessThanUnsigned => (self.get(rs1) < self.get(rs2)) as u32,
                        IntOpMode::ExclusiveOr => self.get(rs1) ^ self.get(rs2),
                        IntOpMode::Or => self.get(rs1) | self.get(rs2),
                        IntOpMode::And => self.get(rs1) & self.get(rs2),
                    },
                );
            }
            FullInstruction::IntShift(IntShift { rd, rs1, rs2, mode }) => {
                if rd == 0 {
                    return;
                }

                self.set(
                    rd,
                    match mode {
                        ShiftMode::LogicalLeft => self.get(rs1) << (self.get(rs2) & 0b11111),
                        ShiftMode::LogicalRight => self.get(rs1) >> (self.get(rs2) & 0b11111),
                        ShiftMode::ArithmeticRight => {
                            ((self.get(rs1) as i32) >> (self.get(rs2) & 0b11111)) as u32
                        }
                    },
                );
            }
            FullInstruction::Fence(_rd, _rs1, _fm, _pred, _succ) => unimplemented!(),
            FullInstruction::ECall => unimplemented!(),
            FullInstruction::EBreak => unimplemented!(),
            // Zifencei
            FullInstruction::FenceI(_rd, _rs1, _immediate) => unimplemented!(),
            // Zicsr
            FullInstruction::CSR(CSR {
                rd: _,
                source: _,
                mode: _,
                csr: _,
            }) => unimplemented!(),
            // RV32M
            FullInstruction::MulOp(MulOp { rd, rs1, rs2, mode }) => {
                if rd == 0 {
                    return;
                }

                self.set(
                    rd,
                    match mode {
                        MulOpMode::Multiply => {
                            (self.get(rs1) as i32).wrapping_mul(self.get(rs2) as i32) as u32
                        }
                        MulOpMode::MultiplyHull => {
                            ((self.get(rs1) as i32 as i64).wrapping_mul(self.get(rs2) as i32 as i64)
                                as u64
                                >> 32) as u32
                        }
                        MulOpMode::MultiplyHullSignedUnsigned => {
                            ((self.get(rs1) as i32 as i64).wrapping_mul(self.get(rs2) as u64 as i64)
                                as u64
                                >> 32) as u32
                        }
                        MulOpMode::MultiplyHullUnsigned => {
                            ((self.get(rs1) as u64).wrapping_mul(self.get(rs2) as u64) >> 32) as u32
                        }
                        MulOpMode::Divide => {
                            (self.get(rs1) as i32).wrapping_div(self.get(rs2) as i32) as u32
                        }
                        MulOpMode::DivideUnsigned => self.get(rs1).wrapping_div(self.get(rs2)),
                        MulOpMode::Remainder => {
                            (self.get(rs1) as i32).wrapping_rem(self.get(rs2) as i32) as u32
                        }
                        MulOpMode::RemainderUnsigned => self.get(rs1).wrapping_rem(self.get(rs2)),
                    },
                );
            }
            // RV32A
            FullInstruction::Atomic(Atomic {
                rd: _,
                rs1: _,
                rs2: _,
                ordering: _,
                mode,
            }) => match mode {
                AtomicMode::LoadReservedWord => unimplemented!(),
                AtomicMode::StoreConditionalWord => unimplemented!(),
                AtomicMode::SwapWord => unimplemented!(),
                AtomicMode::AddWord => unimplemented!(),
                AtomicMode::ExclusiveOrWord => unimplemented!(),
                AtomicMode::AndWord => unimplemented!(),
                AtomicMode::OrWord => unimplemented!(),
                AtomicMode::MinimumWord => unimplemented!(),
                AtomicMode::MaximumWord => unimplemented!(),
                AtomicMode::MinimumUnsignedWord => unimplemented!(),
                AtomicMode::MaximumUnsignedWord => unimplemented!(),
            },
            // RV32F/D
            FullInstruction::FPFusedMultiplyOp(FPFusedMultiplyOp {
                rd,
                rs1,
                rs2,
                rs3,
                add,
                positive,
                rounding,
                precision,
            }) => match precision {
                FPPrecision::Single => {
                    let mut adder = self.get_f32(rs3);
                    if (!add && positive) || (!positive && add) {
                        adder = -adder;
                    }

                    let mut output = self.get_f32(rs1).mul_add(self.get_f32(rs2), adder);
                    if !positive {
                        output = -output;
                    }

                    self.set_f32(rd, self.round_f32(rounding, false, output));
                }
                FPPrecision::Double => {
                    let mut adder = self.get_f64(rs3);
                    if (!add && positive) || (!positive && add) {
                        adder = -adder;
                    }

                    let mut output = self.get_f64(rs1).mul_add(self.get_f64(rs2), adder);
                    if !positive {
                        output = -output;
                    }

                    self.set_f64(rd, self.round_f64(rounding, false, output));
                }
            },
            // RV32F
            FullInstruction::FPSingleOp(FPSingleOp {
                rd,
                rs1,
                rs2,
                mode,
                rounding,
                ret,
            }) => match ret {
                FPReturnMode::Double => unreachable!(),
                FPReturnMode::Single => {
                    self.set_f32(
                        rd,
                        match mode {
                            FPSingleOpMode::Add => self.round_f32(
                                rounding,
                                false,
                                self.get_f32(rs1) + self.get_f32(rs2),
                            ),
                            FPSingleOpMode::Subtract => self.round_f32(
                                rounding,
                                false,
                                self.get_f32(rs1) - self.get_f32(rs2),
                            ),
                            FPSingleOpMode::Multiply => self.round_f32(
                                rounding,
                                false,
                                self.get_f32(rs1) * self.get_f32(rs2),
                            ),
                            FPSingleOpMode::Divide => self.round_f32(
                                rounding,
                                false,
                                self.get_f32(rs1) / self.get_f32(rs2),
                            ),
                            FPSingleOpMode::SquareRoot => {
                                self.round_f32(rounding, false, self.get_f32(rs1).sqrt())
                            }
                            FPSingleOpMode::SignInject => {
                                self.get_f32(rs1).copysign(self.get_f32(rs2))
                            }
                            FPSingleOpMode::SignInjectNot => {
                                self.get_f32(rs1).copysign(-self.get_f32(rs2))
                            }
                            FPSingleOpMode::SignInjectExclusiveOr => {
                                let rs1_bits = self.get_f32(rs1).to_bits();
                                let sign = (rs1_bits >> 31) ^ (self.get_f32(rs2).to_bits() >> 31);
                                f32::from_bits((sign << 31) | ((rs1_bits << 1) >> 1))
                            }
                            FPSingleOpMode::Minimum => self.get_f32(rs1).min(self.get_f32(rs2)),
                            FPSingleOpMode::Maximum => self.get_f32(rs1).max(self.get_f32(rs2)),
                            FPSingleOpMode::ConvertSingleFromWord => {
                                self.round_f32(rounding, false, self.get(rs1) as i32 as f32)
                            }
                            FPSingleOpMode::ConvertSingleFromUnsignedWord => {
                                self.round_f32(rounding, false, self.get(rs1) as f32)
                            }
                            FPSingleOpMode::MoveSingleFromWord => f32::from_bits(self.get(rs1)),
                            _ => unreachable!(),
                        },
                    );
                }
                FPReturnMode::Integer => {
                    self.set(
                        rd,
                        match mode {
                            FPSingleOpMode::Equals => {
                                (self.get_f32(rs1) == self.get_f32(rs2)) as u32
                            }
                            FPSingleOpMode::LessThan => {
                                (self.get_f32(rs1) < self.get_f32(rs2)) as u32
                            }
                            FPSingleOpMode::LessThanOrEqual => {
                                (self.get_f32(rs1) <= self.get_f32(rs2)) as u32
                            }
                            FPSingleOpMode::Class => {
                                let float = self.get_f32(rs1);
                                let mut class = 0u32;

                                if float.is_nan() {
                                    let bits = float.to_bits();
                                    let signal = (bits >> 22) & 1;
                                    if signal == 1 {
                                        class |= 1 << 8;
                                    } else {
                                        class |= 1 << 9;
                                    }
                                } else if float.is_sign_positive() {
                                    if float == 0.0 {
                                        class |= 1 << 4;
                                    } else if float.is_subnormal() {
                                        class |= 1 << 5;
                                    } else if float.is_normal() {
                                        class |= 1 << 6;
                                    } else if float == f32::INFINITY {
                                        class |= 1 << 7;
                                    }
                                } else if float == -0.0 {
                                    class |= 1 << 3;
                                } else if float.is_subnormal() {
                                    class |= 1 << 2;
                                } else if float.is_normal() {
                                    class |= 1 << 1;
                                } else if float == f32::NEG_INFINITY {
                                    class |= 1;
                                }

                                class
                            }
                            FPSingleOpMode::ConvertWordFromSingle => {
                                self.round_f32(rounding, false, self.get_f32(rs1)) as i32 as u32
                            }
                            FPSingleOpMode::ConvertUnsignedWordFromSingle => {
                                self.round_f32(rounding, false, self.get_f32(rs1)) as u32
                            }
                            FPSingleOpMode::MoveWordFromSingle => self.get_f32(rs1).to_bits(),
                            _ => unreachable!(),
                        },
                    );
                }
            },
            // RV32D
            FullInstruction::FPDoubleOp(FPDoubleOp {
                rd,
                rs1,
                rs2,
                mode,
                rounding,
                ret,
            }) => match ret {
                FPReturnMode::Double => {
                    self.set_f64(
                        rd,
                        match mode {
                            FPDoubleOpMode::Add => self.round_f64(
                                rounding,
                                false,
                                self.get_f64(rs1) + self.get_f64(rs2),
                            ),
                            FPDoubleOpMode::Subtract => self.round_f64(
                                rounding,
                                false,
                                self.get_f64(rs1) - self.get_f64(rs2),
                            ),
                            FPDoubleOpMode::Multiply => self.round_f64(
                                rounding,
                                false,
                                self.get_f64(rs1) * self.get_f64(rs2),
                            ),
                            FPDoubleOpMode::Divide => self.round_f64(
                                rounding,
                                false,
                                self.get_f64(rs1) / self.get_f64(rs2),
                            ),
                            FPDoubleOpMode::SquareRoot => {
                                self.round_f64(rounding, false, self.get_f64(rs1).sqrt())
                            }
                            FPDoubleOpMode::SignInject => {
                                self.get_f64(rs1).copysign(self.get_f64(rs2))
                            }
                            FPDoubleOpMode::SignInjectNot => {
                                self.get_f64(rs1).copysign(-self.get_f64(rs2))
                            }
                            FPDoubleOpMode::SignInjectExclusiveOr => {
                                let rs1_bits = self.get_f64(rs1).to_bits();
                                let sign = (rs1_bits >> 63) ^ (self.get_f64(rs2).to_bits() >> 63);
                                f64::from_bits((sign << 63) | ((rs1_bits << 1) >> 1))
                            }
                            FPDoubleOpMode::Minimum => self.get_f64(rs1).min(self.get_f64(rs2)),
                            FPDoubleOpMode::Maximum => self.get_f64(rs1).max(self.get_f64(rs2)),
                            FPDoubleOpMode::ConvertDoubleFromSingle => self.get_f32(rs1) as f64,
                            FPDoubleOpMode::ConvertDoubleFromWord => {
                                self.round_f64(rounding, false, self.get(rs1) as i32 as i64 as f64)
                            }
                            FPDoubleOpMode::ConvertDoubleFromUnsignedWord => {
                                self.round_f64(rounding, false, self.get(rs1) as u64 as f64)
                            }
                            _ => unreachable!(),
                        },
                    );
                }
                FPReturnMode::Single => {
                    self.set_f32(
                        rd,
                        match mode {
                            FPDoubleOpMode::ConvertSingleFromDouble => {
                                self.round_f32(rounding, false, self.get_f64(rs1) as f32)
                            }
                            _ => unreachable!(),
                        },
                    );
                }
                FPReturnMode::Integer => {
                    self.set(
                        rd,
                        match mode {
                            FPDoubleOpMode::Equals => {
                                (self.get_f64(rs1) == self.get_f64(rs2)) as u32
                            }
                            FPDoubleOpMode::LessThan => {
                                (self.get_f64(rs1) < self.get_f64(rs2)) as u32
                            }
                            FPDoubleOpMode::LessThanOrEqual => {
                                (self.get_f64(rs1) <= self.get_f64(rs2)) as u32
                            }
                            FPDoubleOpMode::Class => {
                                let float = self.get_f64(rs1);
                                let mut class: u32 = 0;

                                if float.is_nan() {
                                    let bits = float.to_bits();
                                    let signal = (bits >> 22) & 1;
                                    if signal == 1 {
                                        class |= 1 << 8;
                                    } else {
                                        class |= 1 << 9;
                                    }
                                } else if float.is_sign_positive() {
                                    if float == 0.0 {
                                        class |= 1 << 4;
                                    } else if float.is_subnormal() {
                                        class |= 1 << 5;
                                    } else if float.is_normal() {
                                        class |= 1 << 6;
                                    } else if float == f64::INFINITY {
                                        class |= 1 << 7;
                                    }
                                } else if float == -0.0 {
                                    class |= 1 << 3;
                                } else if float.is_subnormal() {
                                    class |= 1 << 2;
                                } else if float.is_normal() {
                                    class |= 1 << 1;
                                } else if float == f64::NEG_INFINITY {
                                    class |= 1;
                                }

                                class
                            }
                            FPDoubleOpMode::ConvertWordFromDouble => {
                                self.round_f64(rounding, false, self.get_f64(rs1)) as i64 as i32
                                    as u32
                            }
                            FPDoubleOpMode::ConvertUnsignedWordFromDouble => {
                                self.round_f64(rounding, false, self.get_f64(rs1)) as u64 as u32
                            }
                            _ => unreachable!(),
                        },
                    );
                }
            },
        }

        self.pipeline.get_mut(2).unwrap().stage = Stage::Memory;
    }

    pub fn memory(&mut self) {
        if let Some(op) = &self.pipeline[3].memop {
            let bus = BUS.read();
            let mem = bus.dram.memory.memory();

            match op {
                MemoryOperation::Load(address, mode) => {
                    let m = *mode;
                    let size = m.size();
                    let mut buf = [0; 8];
                    let res = mem
                        .read(&mut buf[0..size], GuestAddress(*address as u64))
                        .unwrap();
                    if res != size {
                        #[cfg(feature = "debug")]
                        println!(" | panicked!");
                        panic!("out of bounds memory access");
                    }

                    if let Some(rd) = &mut self.pipeline[3].rd {
                        match rd {
                            ReturnMode::Integer(_, out) => {
                                *out = match m {
                                    LoadMode::Byte => sign_extend_u32(buf[0] as u32, 8) as u32,
                                    LoadMode::HalfWord => sign_extend_u32(
                                        u16::from_le_bytes(buf[0..2].try_into().unwrap()) as u32,
                                        16,
                                    )
                                        as u32,
                                    LoadMode::Word => {
                                        u32::from_le_bytes(buf[0..4].try_into().unwrap())
                                    }
                                    LoadMode::UnsignedByte => buf[0] as u32,
                                    LoadMode::UnsignedHalfWord => {
                                        u16::from_le_bytes(buf[0..2].try_into().unwrap()) as u32
                                    }
                                    _ => unreachable!(),
                                }
                            }
                            ReturnMode::Single(_, out) => {
                                *out = f32::from_le_bytes(buf[0..4].try_into().unwrap())
                            }
                            ReturnMode::Double(_, out) => *out = f64::from_le_bytes(buf),
                        }
                    } else {
                        panic!("load w/o return");
                    }
                }
                MemoryOperation::Store(address, bytes) => {
                    let size = bytes.len();
                    let res = mem
                        .write(&bytes[0..size], GuestAddress(*address as u64))
                        .unwrap();
                    if res != size {
                        #[cfg(feature = "debug")]
                        println!(" | panicked!");
                        panic!("out of bounds memory access");
                    }
                }
            }
        }

        self.pipeline.get_mut(3).unwrap().stage = Stage::Write;
    }

    pub fn write(&mut self) {
        let item = &self.pipeline[4];

        if let Some(rd) = &item.rd {
            match rd {
                ReturnMode::Integer(r, value) => {
                    if r != &0 {
                        self.x[*r] = *value
                    }
                }
                ReturnMode::Single(r, value) => {
                    self.f[*r] = f64::from_bits(0xFFFFFFFF00000000u64 | value.to_bits() as u64)
                }
                ReturnMode::Double(r, value) => self.f[*r] = *value,
            }

            #[cfg(feature = "debug")]
            {
                let signed_x: [i32; 32] = unsafe { transmute(self.x) };
                println!("x: {:?}", signed_x);
            }
        }

        self.pipeline.remove(4);
    }

    // Perform rounding based on RM table. (f32)
    #[inline]
    pub fn round_f32(&self, rm: FPRoundingMode, frm: bool, value: f32) -> f32 {
        rm.apply_f32((self.csrs[0x003] >> 5) & 0b111, frm, value)
    }

    // Perform rounding based on RM table. (f64)
    #[inline]
    pub fn round_f64(&self, rm: FPRoundingMode, frm: bool, value: f64) -> f64 {
        rm.apply_f64((self.csrs[0x003] >> 5) & 0b111, frm, value)
    }

    // Pull a NaN-boxed f32 from f and NaN-unbox it.
    #[inline]
    pub fn get_f32(&self, reg: usize) -> f32 {
        f32::from_bits((self.f[reg].to_bits() & 0xFFFFFFFF) as u32)
    }

    #[inline]
    pub fn get_f64(&self, reg: usize) -> f64 {
        self.f[reg]
    }

    #[inline]
    pub fn get(&self, reg: usize) -> u32 {
        if reg == 0 {
            return 0;
        }

        self.x[reg]
    }

    // NaN-box an f32 and place it in f.
    #[inline]
    pub fn set_f32(&mut self, reg: usize, value: f32) {
        self.pipeline[2].rd = Some(ReturnMode::Single(reg, value));
    }

    #[inline]
    pub fn set_f64(&mut self, reg: usize, value: f64) {
        self.pipeline[2].rd = Some(ReturnMode::Double(reg, value));
    }

    #[inline]
    pub fn set(&mut self, reg: usize, value: u32) {
        if reg == 0 {
            return;
        }

        self.pipeline[2].rd = Some(ReturnMode::Integer(reg, value));
    }
}
