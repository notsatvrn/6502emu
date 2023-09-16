use crate::instructions::riscv32::*;
use crate::io::{Region, BUS};
use ahash::AHashSet;
use parking_lot::RwLock;
use std::sync::LazyLock;
use vm_memory::{Bytes, GuestAddress, GuestAddressSpace, GuestMemory, GuestMemoryRegion};

pub static HARTS: LazyLock<RwLock<Vec<Hart>>> = LazyLock::new(|| RwLock::new(Vec::new()));

pub static RUNNING_HARTS: LazyLock<RwLock<AHashSet<usize>>> =
    LazyLock::new(|| RwLock::new(AHashSet::new()));

pub fn init(hart_count: usize, memory_size: u32) {
    BUS.write().dram.size = memory_size as u64;

    let mut csrs = [0u32; 4096];

    // machine info registers
    // mvendorid (0xF11) is already 0, keep because this is a non-commercial implementation
    csrs[0xF12] = 35; // marchid

    // machine trap setup
    csrs[0x301] = 0b01000000000000000001000100101100; // misa

    let mut i: usize = 0;
    let mut harts = HARTS.write();
    while i < hart_count {
        // hart-specific csrs
        csrs[0xF14] = i as u32; // mhartid

        harts.push(Hart {
            region: Region(0, 0),
            x: [0; 32],
            f: [0.0; 32],
            csrs,
            pc: 0,
        });

        i += 1;
    }

    _ = *RUNNING_HARTS; // initialize
}

// replace this whole thing with a proper MMU
pub fn prepare_hart(hart_id: usize, data: Vec<u8>) {
    let mut harts = HARTS.write();
    let mut bus = BUS.write();
    let mut running_harts = RUNNING_HARTS.write();

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
    let hart = harts.get_mut(hart_id).unwrap();
    hart.region = Region(address, data_len as usize);

    bus.dram.add_region(&hart.region);

    let atomic_memory = bus.dram.memory.memory();
    atomic_memory
        .write(data.as_slice(), GuestAddress(address))
        .unwrap();

    // step 3: declare hart as running
    running_harts.insert(hart_id);
}

#[derive(Clone)]
pub struct Hart {
    pub region: Region,
    pub x: [u32; 32],
    pub f: [f64; 32],
    pub csrs: [u32; 4096],
    pub pc: u32,
}

impl Hart {
    pub fn cycle(&mut self) {
        let index = self.region.0 + self.pc as u64;
        println!(
            "start {} | pc: {} | index: {}",
            self.region.0, self.pc, index
        );

        self.x[4] = self.region.0 as u32; // thread pointer

        let bus = BUS.read();
        let mem = bus.dram.memory.memory();
        let mut buf = [0u8; 4];
        let res = mem.read(&mut buf, GuestAddress(index)).unwrap();
        drop(bus);

        if buf[0] & 0b11 == 0b11 {
            if res != 4 {
                panic!("reached end of program early");
            }

            self.pc += 4;
            let inst_u32 = u32::from_le_bytes(buf);
            let inst = decode_full(inst_u32);
            self.execute_parsed(inst);
            //self.execute(inst_u32);
        } else {
            if res != 2 {
                panic!("reached end of program early");
            }

            self.pc += 2;
            let inst_u16 = u16::from_le_bytes([buf[0], buf[1]]);
            let inst = decode_compressed(inst_u16);
            self.compressed(inst_u16);
        }

        self.x[0] = 0; // reset zero register

        // reset hart if execution finished
        if self.pc as usize >= self.region.1 {
            BUS.write().dram.remove_region(&self.region);
            self.region = Region(0, 0);
            self.x = [0; 32];
            self.f = [0.0; 32];
            self.csrs = [0; 4096];
            self.pc = 0;
        }
    }

    // Helper functions for reading/writing memory.

    #[inline]
    fn read_memory(&mut self, buf: &mut [u8], base: u32, offset: i32) {
        let address = ((base as i32).wrapping_add(offset) as u32 + self.region.0 as u32) as u64;

        let bus = BUS.read();
        let mem = bus.dram.memory.memory();
        let res = mem.read(buf, GuestAddress(address)).unwrap();

        if res != buf.len() {
            println!(" | panicked!");
            panic!("out of bounds memory access");
        }
    }

    #[inline]
    fn write_memory(&mut self, bytes: &[u8], base: u32, offset: i32) {
        let address = ((base as i32).wrapping_add(offset) as u32 + self.region.0 as u32) as u64;

        let bus = BUS.read();
        let mem = bus.dram.memory.memory();
        let res = mem.write(bytes, GuestAddress(address)).unwrap();

        if res != bytes.len() {
            println!(" | panicked!");
            panic!("out of bounds memory access");
        }
    }

    // Load/store instructions which have compressed versions as well.

    #[inline]
    fn lw(&mut self, rd: usize, offset: i32, rs1: usize) {
        let mut buf = [0u8; 4];
        self.read_memory(&mut buf, self.x[rs1], offset);
        self.x[rd] = u32::from_le_bytes(buf);
    }

    #[inline]
    fn flw(&mut self, rd: usize, offset: i32, rs1: usize) {
        let mut buf = [0u8; 4];
        self.read_memory(&mut buf, self.x[rs1], offset);
        self.set_f32(rd, f32::from_le_bytes(buf));
    }

    #[inline]
    fn fld(&mut self, rd: usize, offset: i32, rs1: usize) {
        let mut buf = [0u8; 8];
        self.read_memory(&mut buf, self.x[rs1], offset);
        self.f[rd] = f64::from_le_bytes(buf);
    }

    #[inline]
    fn sw(&mut self, rs1: usize, rs2: usize, offset: i32) {
        self.write_memory(&self.x[rs2].to_le_bytes(), self.x[rs1], offset);
    }

    #[inline]
    fn fsw(&mut self, rs1: usize, rs2: usize, offset: i32) {
        self.write_memory(&self.get_f32(rs2).to_le_bytes(), self.x[rs1], offset);
    }

    #[inline]
    fn fsd(&mut self, rs1: usize, rs2: usize, offset: i32) {
        self.write_memory(&self.f[rs2].to_le_bytes(), self.x[rs1], offset);
    }

    // Execute a compressed instruction.

    pub fn compressed(&mut self, inst: u16) {
        if inst == 0 {
            panic!("invalid instruction");
        }

        let opcode = inst & 0b11;
        let mut rd = ((inst >> 7) & 0b11111) as usize;
        let rs2 = ((inst >> 2) & 0b11111) as usize;
        let funct3 = inst >> 13;

        match opcode {
            0 => match funct3 {
                0b000 => {
                    print!("c.addi4spn");
                    unimplemented!();
                }
                0b001 => {
                    let imm = (((inst >> 5) & 0b11) << 3) | (((inst >> 10) & 0b111) << 1);
                    let offset = imm * 8;
                    self.fld(rs2 & 0b111, offset as i32, rd & 0b111);
                }
                0b010 => {
                    let imm = (((inst >> 5) & 1) << 4)
                        | (((inst >> 10) & 0b111) << 1)
                        | ((inst >> 6) & 1);
                    let offset = imm * 4;
                    self.lw(rs2 & 0b111, offset as i32, rd & 0b111);
                }
                0b011 => {
                    let imm = (((inst >> 5) & 1) << 4)
                        | (((inst >> 10) & 0b111) << 1)
                        | ((inst >> 6) & 1);
                    let offset = imm * 4;
                    self.flw(rs2 & 0b111, offset as i32, rd & 0b111);
                }
                0b101 => {
                    let imm = (((inst >> 5) & 0b11) << 3) | (((inst >> 10) & 0b111) << 1);
                    let offset = imm as i32 * 8;
                    print!("compressed: c.fsd | ");
                    self.fsd(rs2 & 0b111, rs2, offset as i32);
                }
                0b110 => {
                    let imm = (((inst >> 5) & 1) << 4)
                        | (((inst >> 10) & 0b111) << 1)
                        | ((inst >> 6) & 1);
                    let offset = imm as i32 * 4;
                    print!("compressed: c.sw | ");
                    self.write_memory(&self.x[rs2].to_le_bytes(), self.x[2], offset);
                }
                0b111 => {
                    let imm = (((inst >> 5) & 1) << 4)
                        | (((inst >> 10) & 0b111) << 1)
                        | ((inst >> 6) & 1);
                    let offset = imm as i32 * 4;
                    print!("compressed: c.fsw | ");
                    self.fsw(rs2 & 0b111, rs2, offset as i32);
                }
                _ => panic!("invalid instruction"),
            },
            1 => match funct3 {
                0b000 => {
                    print!("c.addi");
                    unimplemented!();
                }
                0b001 => {
                    print!("c.jal");
                    unimplemented!();
                }
                0b010 => {
                    print!("c.li");
                    unimplemented!();
                }
                0b011 => {
                    if rd == 2 {
                        print!("c.addi16sp");
                        unimplemented!();
                    } else if rd != 0 {
                        print!("c.lui");
                        unimplemented!();
                    } else {
                        print!("c.nop");
                        unimplemented!();
                    }
                }
                0b100 => {
                    rd &= 0b111;
                    let imm5 = (inst >> 12) & 1;
                    let funct2_major = (inst >> 10) & 0b11;
                    let funct2_minor = (inst >> 5) & 0b11;

                    match funct2_major {
                        0b00 => {
                            let shamt5 = (inst >> 12) & 1;
                            let shamt = (shamt5 << 5) | ((inst >> 2) & 0b11111);

                            if shamt5 == 0 && shamt != 0 {
                                print!("c.srli x{}, {}", rd, shamt);
                                self.x[rd] >>= shamt;
                                println!(" | rd: {}", self.x[rd]);
                            } else if shamt == 0 {
                                println!("HINT: RV32/64C");
                            } else if shamt5 != 0 {
                                println!("NSE: RV32C");
                            }
                        }
                        0b01 => {
                            let shamt5 = (inst >> 12) & 1;
                            let shamt = (shamt5 << 5) | ((inst >> 2) & 0b11111);

                            if shamt5 == 0 && shamt != 0 {
                                print!("c.srai x{}, {}", rd, shamt);
                                self.x[rd] = ((self.x[rd] as i32) >> shamt) as u32;
                                println!(" | rd: {}", self.x[rd]);
                            } else if shamt == 0 {
                                println!("HINT: RV32/64C");
                            } else if shamt5 != 0 {
                                println!("NSE: RV32C");
                            }
                        }
                        0b10 => {
                            print!("c.andi");
                            unimplemented!();
                        }
                        0b11 if imm5 == 0 => match funct2_minor {
                            0b00 => {
                                print!("c.sub");
                                unimplemented!();
                            }
                            0b01 => {
                                print!("c.xor");
                                unimplemented!();
                            }
                            0b10 => {
                                print!("c.or");
                                unimplemented!();
                            }
                            0b11 => {
                                print!("c.and");
                                unimplemented!();
                            }
                            _ => panic!("invalid instruction"),
                        },
                        0b11 if imm5 == 1 => match funct2_minor {
                            0b00 => {
                                print!("c.subw");
                                unimplemented!();
                            }
                            0b01 => {
                                print!("c.addw");
                                unimplemented!();
                            }
                            0b10 => {} // reserved
                            0b11 => {} // reserved
                            _ => panic!("invalid instruction"),
                        },
                        _ => panic!("invalid instruction"),
                    }
                }
                0b101 => {
                    print!("c.j");
                    unimplemented!();
                }
                0b110 => {
                    print!("c.beqz");
                    unimplemented!();
                }
                0b111 => {
                    print!("c.bnez");
                    unimplemented!();
                }
                _ => panic!("invalid instruction"),
            },
            2 => match funct3 {
                0b000 => {
                    let shamt5 = (inst >> 12) & 1;
                    let shamt = (shamt5 << 5) | ((inst >> 2) & 0b11111);

                    if shamt5 == 0 && shamt != 0 {
                        print!("c.slli x{}, {}", rd, shamt);
                        self.x[rd] <<= shamt;
                        println!(" | rd: {}", self.x[rd]);
                    } else if shamt == 0 {
                        println!("HINT: RV32/64C");
                    } else if shamt5 != 0 {
                        println!("NSE: RV32C");
                    } else {
                        println!("HINT: all base ISAs");
                    }
                }
                0b001 => {
                    let imm = ((inst >> 5) & 0b11)
                        | (((inst >> 12) & 1) << 2)
                        | (((inst >> 2) & 0b111) << 3);
                    let offset = imm as i32 * 8;
                    print!("compressed: c.fldsp | ");
                    self.fld(rd, offset as i32, 2);
                }
                0b010 => {
                    let imm = ((inst >> 4) & 0b111)
                        | (((inst >> 12) & 1) << 2)
                        | (((inst >> 2) & 0b11) << 3);
                    let offset = imm as i32 * 4;
                    print!("compressed: c.lwsp | ");
                    self.lw(rd, offset as i32, 2);
                }
                0b011 => {
                    let imm = ((inst >> 4) & 0b111)
                        | (((inst >> 12) & 1) << 2)
                        | (((inst >> 2) & 0b11) << 3);
                    let offset = imm as i32 * 4;
                    print!("compressed: c.flwsp | ");
                    self.flw(rd, offset as i32, 2);
                }
                0b100 => {
                    if (inst >> 12) != 0b1001 {
                        if rd != 0 {
                            if rs2 == 0 {
                                print!("c.jr x{}", rd);
                                self.pc = self.x[rd];
                                println!(" | pc: {}", self.pc);
                            } else {
                                print!("c.mv x{}, x{}", rd, rs2);
                                self.x[rd] = self.x[rs2];
                                println!(" | rd: {}", self.x[rd]);
                            }
                        } else {
                            println!("HINTs");
                        }
                    } else if rd != 0 {
                        if rs2 == 0 {
                            print!("c.jalr x{}", rd);
                            self.pc = self.x[rd];
                            self.x[1] = self.pc + 2;
                            println!(" | pc: {}", self.pc);
                        } else {
                            print!("c.add x{}, x{}", rd, rs2);
                            self.x[rd] += self.x[rs2];
                            println!(" | rd: {}", self.x[rd]);
                        }
                    } else if rs2 == 0 {
                        println!("c.ebreak");
                        unimplemented!();
                    } else {
                        println!("HINTs");
                    }
                }
                0b101 => {
                    let imm = (((inst >> 7) & 0b111) << 3) | ((inst >> 10) & 0b111);
                    let offset = imm as i32 * 8;
                    self.fsd(2, rs2, offset as i32);
                }
                0b110 => {
                    let imm = (((inst >> 7) & 0b11) << 4) | ((inst >> 9) & 0b1111);
                    let offset = imm as i32 * 4;
                    self.write_memory(&self.x[rs2].to_le_bytes(), self.x[2], offset);
                }
                0b111 => {
                    let imm = (((inst >> 7) & 0b11) << 4) | ((inst >> 9) & 0b1111);
                    let offset = imm as i32 * 4;
                    self.fsw(2, rs2, offset as i32);
                }
                _ => panic!("invalid instruction"),
            },
            _ => panic!("invalid instruction"),
        }
    }

    pub fn execute_parsed(&mut self, inst: Instruction) {
        print!("{}", inst);

        match inst {
            Instruction::LUI(rd, immediate) => {
                self.x[rd] = (immediate << 12) - 4;
                println!(" | rd: {}", self.x[rd]);
            }
            Instruction::AUIPC(rd, immediate) => {
                self.pc = self.pc.wrapping_add(immediate << 12);
                self.x[rd] = self.pc;
                println!(" | rd: {}", self.x[rd]);
            }
            Instruction::JAL(rd, offset) => {
                self.pc = (self.pc as i32).wrapping_add(offset) as u32 - 4;
                self.x[rd] = self.pc + 4;
                println!(" | rd: {}", self.x[rd]);
            }
            Instruction::JALR(rd, offset, rs1) => {
                self.pc = (((self.x[rs1] as i32).wrapping_add(offset) as u32) << 1) >> 1;
                self.x[rd] = self.pc + 4;
                println!(" | rd: {}", self.x[rd]);
            }
            Instruction::Branch(Branch {
                rs1,
                rs2,
                offset,
                mode,
            }) => {
                let branching = match mode {
                    BranchMode::Equal => (self.x[rs1] as i32) == self.x[rs2] as i32,
                    BranchMode::NotEqual => (self.x[rs1] as i32) != self.x[rs2] as i32,
                    BranchMode::LessThan => (self.x[rs1] as i32) < self.x[rs2] as i32,
                    BranchMode::GreaterOrEqual => (self.x[rs1] as i32) >= self.x[rs2] as i32,
                    BranchMode::LessThanUnsigned => self.x[rs1] < self.x[rs2],
                    BranchMode::GreaterOrEqualUnsigned => self.x[rs1] >= self.x[rs2],
                };

                if branching {
                    self.pc = (self.pc as i32).wrapping_add(offset) as u32;
                    println!(" | branched");
                } else {
                    println!(" | didn't branch");
                }
            }
            Instruction::Load(Load {
                rd,
                rs1,
                offset,
                mode,
            }) => {
                match mode {
                    LoadMode::Byte => {
                        let mut buf = [0u8; 1];
                        self.read_memory(&mut buf, self.x[rs1], offset);
                        self.x[rd] = sign_extend(buf[0] as u32, 8) as u32;
                    }
                    LoadMode::HalfWord => {
                        let mut buf = [0u8; 2];
                        self.read_memory(&mut buf, self.x[rs1], offset);
                        self.x[rd] = sign_extend(u16::from_le_bytes(buf) as u32, 16) as u32;
                    }
                    LoadMode::Word => self.lw(rd, offset, rs1),
                    LoadMode::UnsignedByte => {
                        let mut buf = [0u8; 1];
                        self.read_memory(&mut buf, self.x[rs1], offset);
                        self.x[rd] = buf[0] as u32;
                    }
                    LoadMode::UnsignedHalfWord => {
                        let mut buf = [0u8; 2];
                        self.read_memory(&mut buf, self.x[rs1], offset);
                        self.x[rd] = u16::from_le_bytes(buf) as u32;
                    }
                }

                println!(" | rd: {}", self.x[rd] as i32);
            }
            Instruction::Store(Store {
                rs1,
                rs2,
                offset,
                mode,
            }) => {
                match mode {
                    StoreMode::Byte => {
                        self.write_memory(&(self.x[rs2] as u8).to_le_bytes(), self.x[rs1], offset);
                    }
                    StoreMode::HalfWord => {
                        self.write_memory(&(self.x[rs2] as u16).to_le_bytes(), self.x[rs1], offset);
                    }
                    StoreMode::Word => {
                        self.sw(rs1, rs2, offset);
                    }
                }

                println!(" | value: {}", self.x[rs2] as i32);
            }
            Instruction::IMMOp(IMMOp {
                rd,
                rs1,
                immediate,
                mode,
            }) => {
                self.x[rd] = match mode {
                    IMMOpMode::Add => (self.x[rs1] as i32).wrapping_add(immediate) as u32,
                    IMMOpMode::SetLessThan => ((self.x[rs1] as i32) < immediate) as u32,
                    IMMOpMode::SetLessThanUnsigned => (self.x[rs1] < immediate as u32) as u32,
                    IMMOpMode::ExclusiveOr => self.x[rs1] ^ (immediate as u32),
                    IMMOpMode::Or => self.x[rs1] | (immediate as u32),
                    IMMOpMode::And => self.x[rs1] & (immediate as u32),
                };

                println!(" | rd: {}", self.x[rd] as i32);
            }
            Instruction::IMMShift(IMMShift {
                rd,
                rs1,
                amount,
                mode,
            }) => {
                self.x[rd] = match mode {
                    ShiftMode::LogicalLeft => self.x[rs1] << amount,
                    ShiftMode::LogicalRight => self.x[rs1] >> amount,
                    ShiftMode::ArithmeticRight => ((self.x[rs1] as i32) >> amount) as u32,
                };

                println!(" | rd: {}", self.x[rd] as i32);
            }
            Instruction::IntOp(IntOp { rd, rs1, rs2, mode }) => {
                self.x[rd] = match mode {
                    IntOpMode::Add => self.x[rs1].wrapping_add(self.x[rs2]),
                    IntOpMode::Subtract => self.x[rs1].wrapping_sub(self.x[rs2]),
                    IntOpMode::SetLessThan => ((self.x[rs1] as i32) < (self.x[rs2] as i32)) as u32,
                    IntOpMode::SetLessThanUnsigned => (self.x[rs1] < self.x[rs2]) as u32,
                    IntOpMode::ExclusiveOr => self.x[rs1] ^ self.x[rs2],
                    IntOpMode::Or => self.x[rs1] | self.x[rs2],
                    IntOpMode::And => self.x[rs1] & self.x[rs2],
                };

                println!(" | rd: {}", self.x[rd] as i32);
            }
            Instruction::IntShift(IntShift { rd, rs1, rs2, mode }) => {
                self.x[rd] = match mode {
                    ShiftMode::LogicalLeft => self.x[rs1] << (self.x[rs2] & 0b11111),
                    ShiftMode::LogicalRight => self.x[rs1] >> (self.x[rs2] & 0b11111),
                    ShiftMode::ArithmeticRight => {
                        ((self.x[rs1] as i32) >> (self.x[rs2] & 0b11111)) as u32
                    }
                };

                println!(" | rd: {}", self.x[rd] as i32);
            }
            Instruction::Fence(rd, rs1, fm, pred, succ) => unimplemented!(),
            Instruction::ECall => unimplemented!(),
            Instruction::EBreak => unimplemented!(),
            // Zifencei
            Instruction::FenceI(rd, rs1, immediate) => unimplemented!(),
            // Zicsr
            Instruction::CSR(CSR {
                rd,
                source,
                mode,
                csr,
            }) => unimplemented!(),
            // RV32M
            Instruction::MulOp(MulOp { rd, rs1, rs2, mode }) => {
                self.x[rd] = match mode {
                    MulOpMode::Multiply => {
                        (self.x[rs1] as i32).wrapping_mul(self.x[rs2] as i32) as u32
                    }
                    MulOpMode::MultiplyHull => {
                        ((self.x[rs1] as i32 as i64).wrapping_mul(self.x[rs2] as i32 as i64) as u64
                            >> 32) as u32
                    }
                    MulOpMode::MultiplyHullSignedUnsigned => {
                        ((self.x[rs1] as i32 as i64).wrapping_mul(self.x[rs2] as u64 as i64) as u64
                            >> 32) as u32
                    }
                    MulOpMode::MultiplyHullUnsigned => {
                        ((self.x[rs1] as u64).wrapping_mul(self.x[rs2] as u64) >> 32) as u32
                    }
                    MulOpMode::Divide => {
                        (self.x[rs1] as i32).wrapping_div(self.x[rs2] as i32) as u32
                    }
                    MulOpMode::DivideUnsigned => self.x[rs1].wrapping_div(self.x[rs2]),
                    MulOpMode::Remainder => {
                        (self.x[rs1] as i32).wrapping_rem(self.x[rs2] as i32) as u32
                    }
                    MulOpMode::RemainderUnsigned => (self.x[rs1]).wrapping_rem(self.x[rs2]),
                };

                println!(" | rd: {}", self.x[rd] as i32);
            }
            // RV32A
            Instruction::Atomic(Atomic {
                rd,
                rs1,
                rs2,
                ordering,
                mode,
            }) => unimplemented!(),
            // RV32F/D
            Instruction::FPLoad(FPLoad {
                rd,
                rs1,
                offset,
                precision,
            }) => match precision {
                FPPrecision::Single => self.flw(rd, offset, rs1),
                FPPrecision::Double => self.fld(rd, offset, rs1),
            },
            Instruction::FPStore(FPStore {
                rs1,
                rs2,
                offset,
                precision,
            }) => match precision {
                FPPrecision::Single => self.fsw(rs1, rs2, offset),
                FPPrecision::Double => self.fsd(rs1, rs2, offset),
            },
            Instruction::FPFusedMultiplyOp(FPFusedMultiplyOp {
                rd,
                rs1,
                rs2,
                rs3,
                add,
                positive,
                precision,
            }) => unimplemented!(),
            // RV32F
            Instruction::FPSingleOp(FPSingleOp { rd, rs1, rs2, mode }) => unimplemented!(),
            // RV32D
            Instruction::FPDoubleOp(FPDoubleOp { rd, rs1, rs2, mode }) => unimplemented!(),
        }
    }

    // Execute an uncompressed instruction.

    pub fn execute(&mut self, inst: u32) {
        let opcode = inst & 0b1111111;
        let rd = ((inst >> 7) & 0b11111) as usize;
        let funct3 = (inst >> 12) & 0b111;
        let rs1 = ((inst >> 15) & 0b11111) as usize;
        let rs2 = ((inst >> 20) & 0b11111) as usize;
        let funct7 = inst >> 25;
        let funct2 = funct7 & 0b11;
        let rs3 = (inst >> 27) as usize;

        match opcode {
            0b1000011 if funct2 == 0 => {
                print!("fmadd.s x{}, x{}, x{}, x{}", rd, rs1, rs2, rs3);
                self.set_f32(
                    rd,
                    self.round_rm_f32(
                        funct3,
                        false,
                        self.get_f32(rs1)
                            .mul_add(self.get_f32(rs2), self.get_f32(rs3)),
                    ),
                );
                println!(" | rd: {}", self.get_f32(rd));
            }
            0b1000011 if funct2 == 1 => {
                print!("fmadd.d x{}, x{}, x{}, x{}", rd, rs1, rs2, rs3);
                self.f[rd] =
                    self.round_rm_f64(funct3, false, self.f[rs1].mul_add(self.f[rs2], self.f[rs3]));
                println!(" | rd: {}", self.f[rd]);
            }
            0b1000111 if funct2 == 0 => {
                print!("fmsub.s x{}, x{}, x{}, x{}", rd, rs1, rs2, rs3);
                self.set_f32(
                    rd,
                    self.round_rm_f32(
                        funct3,
                        false,
                        self.get_f32(rs1)
                            .mul_add(self.get_f32(rs2), -self.get_f32(rs3)),
                    ),
                );
                println!(" | rd: {}", self.get_f32(rd));
            }
            0b1000111 if funct2 == 1 => {
                print!("fmsub.d x{}, x{}, x{}, x{}", rd, rs1, rs2, rs3);
                self.f[rd] = self.round_rm_f64(
                    funct3,
                    false,
                    self.f[rs1].mul_add(self.f[rs2], -self.f[rs3]),
                );
                println!(" | rd: {}", self.f[rd]);
            }
            0b1001011 if funct2 == 0 => {
                print!("fnmsub.s x{}, x{}, x{}, x{}", rd, rs1, rs2, rs3);
                self.set_f32(
                    rd,
                    self.round_rm_f32(
                        funct3,
                        false,
                        -self
                            .get_f32(rs1)
                            .mul_add(self.get_f32(rs2), self.get_f32(rs3)),
                    ),
                );
                println!(" | rd: {}", self.get_f32(rd));
            }
            0b1001011 if funct2 == 1 => {
                print!("fnmsub.d x{}, x{}, x{}, x{}", rd, rs1, rs2, rs3);
                self.f[rd] = self.round_rm_f64(
                    funct3,
                    false,
                    -self.f[rs1].mul_add(self.f[rs2], self.f[rs3]),
                );
                println!(" | rd: {}", self.f[rd]);
            }
            0b1001111 if funct2 == 0 => {
                print!("fnmadd.s x{}, x{}, x{}, x{}", rd, rs1, rs2, rs3);
                self.set_f32(
                    rd,
                    self.round_rm_f32(
                        funct3,
                        false,
                        -self
                            .get_f32(rs1)
                            .mul_add(self.get_f32(rs2), -self.get_f32(rs3)),
                    ),
                );
                println!(" | rd: {}", self.get_f32(rd));
            }
            0b1001111 if funct2 == 1 => {
                print!("fnmadd.d x{}, x{}, x{}, x{}", rd, rs1, rs2, rs3);
                self.f[rd] = self.round_rm_f64(
                    funct3,
                    false,
                    -self.f[rs1].mul_add(self.f[rs2], -self.f[rs3]),
                );
                println!(" | rd: {}", self.f[rd]);
            }
            0b1010011 => match funct7 {
                // RV32F
                0b0000000 => self.set_f32(
                    rd,
                    self.round_rm_f32(funct3, false, self.get_f32(rs1) + self.get_f32(rs2)),
                ),
                0b0000100 => self.set_f32(
                    rd,
                    self.round_rm_f32(funct3, false, self.get_f32(rs1) - self.get_f32(rs2)),
                ),
                0b0001000 => self.set_f32(
                    rd,
                    self.round_rm_f32(funct3, false, self.get_f32(rs1) * self.get_f32(rs2)),
                ),
                0b0001100 => self.set_f32(
                    rd,
                    self.round_rm_f32(funct3, false, self.get_f32(rs1) / self.get_f32(rs2)),
                ),
                0b0101100 if rs2 == 0b00000 => self.set_f32(
                    rd,
                    self.round_rm_f32(funct3, false, self.get_f32(rs1).sqrt()),
                ),
                0b0010000 => match funct3 {
                    0 => self.set_f32(rd, self.get_f32(rs1).copysign(self.get_f32(rs2))),
                    1 => self.set_f32(rd, self.get_f32(rs1).copysign(-self.get_f32(rs2))),
                    2 => {
                        let sign = (self.get_f32(rs1).to_bits() >> 31)
                            ^ (self.get_f32(rs2).to_bits() >> 31);
                        self.set_f32(
                            rd,
                            f32::from_bits(
                                (sign << 31) | ((self.get_f32(rs1).to_bits() << 1) >> 1),
                            ),
                        );
                    }
                    _ => panic!("invalid instruction"),
                },
                0b0010100 => match funct3 {
                    0 => self.set_f32(rd, self.get_f32(rs1).min(self.get_f32(rs2))),
                    1 => self.set_f32(rd, self.get_f32(rs1).max(self.get_f32(rs2))),
                    _ => panic!("invalid instruction"),
                },
                0b1100000 => match rs2 {
                    0 => {
                        self.x[rd] =
                            self.round_rm_f32(funct3, false, self.get_f32(rs1)) as i32 as u32
                    }
                    1 => self.x[rd] = self.round_rm_f32(funct3, false, self.get_f32(rs1)) as u32,
                    _ => panic!("invalid instruction"),
                },
                0b1110000 => match funct3 {
                    0 => {
                        self.x[rd] = self.get_f32(rs1).to_bits();
                    }
                    1 => {
                        let float = self.get_f32(rs1);
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

                        self.x[rd] = class;
                    }
                    _ => panic!("invalid instruction"),
                },
                0b1010000 => match funct3 {
                    2 => self.x[rd] = (self.get_f32(rs1) == self.get_f32(rs2)) as u32,
                    1 => self.x[rd] = (self.get_f32(rs1) < self.get_f32(rs2)) as u32,
                    0 => self.x[rd] = (self.get_f32(rs1) <= self.get_f32(rs2)) as u32,
                    _ => panic!("invalid instruction"),
                },
                0b1101000 => match rs2 {
                    0 => self.set_f32(
                        rd,
                        self.round_rm_f32(funct3, false, self.x[rs1] as i32 as f32),
                    ),
                    1 => self.set_f32(rd, self.round_rm_f32(funct3, false, self.x[rs1] as f32)),
                    _ => panic!("invalid instruction"),
                },
                0b1111000 => self.set_f32(rd, f32::from_bits(self.x[rs1])),
                // RV32D
                0b0000001 => {
                    self.f[rd] = self.round_rm_f64(funct3, false, self.f[rs1] + self.f[rs2])
                }
                0b0000101 => {
                    self.f[rd] = self.round_rm_f64(funct3, false, self.f[rs1] - self.f[rs2])
                }
                0b0001001 => {
                    self.f[rd] = self.round_rm_f64(funct3, false, self.f[rs1] * self.f[rs2])
                }
                0b0001101 => {
                    self.f[rd] = self.round_rm_f64(funct3, false, self.f[rs1] / self.f[rs2])
                }
                0b0101101 if rs2 == 0 => {
                    self.f[rd] = self.round_rm_f64(funct3, false, self.f[rs1].sqrt())
                }
                0b0010001 => match funct3 {
                    0 => self.f[rd] = self.f[rs1].copysign(self.f[rs2]),
                    1 => self.f[rd] = self.f[rs1].copysign(-self.f[rs2]),
                    2 => {
                        let sign = (self.f[rs1].to_bits() >> 63) ^ (self.f[rs2].to_bits() >> 63);
                        self.f[rd] =
                            f64::from_bits((sign << 63) | ((self.f[rs1].to_bits() << 1) >> 1));
                    }
                    _ => panic!("invalid instruction"),
                },
                0b0010101 => match funct3 {
                    0 => self.f[rd] = self.f[rs1].min(self.f[rs2]),
                    1 => self.f[rd] = self.f[rs1].max(self.f[rs2]),
                    _ => panic!("invalid instruction"),
                },
                0b0100000 if rs2 == 1 => {
                    print!("fcvt.s.d x{}, x{}", rd, rs1);
                    self.f[rd] = self.round_rm_f64(funct3, false, self.get_f32(rs1) as f64);
                    println!(" | rd: {}", self.f[rd]);
                }
                0b0100001 if rs2 == 0 => {
                    print!("fcvt.d.s x{}, x{}", rd, rs1);
                    self.set_f32(rd, self.round_rm_f32(funct3, false, self.f[rs1] as f32));
                    println!(" | rd: {}", self.get_f32(rd));
                }
                0b1010001 => match funct3 {
                    2 => self.x[rd] = (self.f[rs1] == self.f[rs2]) as u32,
                    1 => self.x[rd] = (self.f[rs1] < self.f[rs2]) as u32,
                    0 => self.x[rd] = (self.f[rs1] <= self.f[rs2]) as u32,
                    _ => panic!("invalid instruction"),
                },
                0b1110001 if rs2 == 0 && funct3 == 1 => {
                    print!("fclass.d x{}, x{}", rd, rs1);
                    let float = self.f[rs1];
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

                    self.x[rd] = class;
                    println!(" | rd: {}", self.x[rd] as i32);
                }
                0b1100001 if rs2 == 0 => {
                    print!("fcvt.w.d x{}, x{}", rd, rs1);
                    self.x[rd] = self.round_rm_f64(funct3, false, self.f[rs1]) as i64 as i32 as u32;
                    println!(" | rd: {}", self.f[rd]);
                }
                0b1100001 if rs2 == 1 => {
                    print!("fcvt.wu.d x{}, x{}", rd, rs1);
                    self.x[rd] = self.round_rm_f64(funct3, false, self.f[rs1]) as u64 as u32;
                    println!(" | rd: {}", self.f[rd]);
                }
                0b1101001 if rs2 == 0 => {
                    print!("fcvt.d.w x{}, x{}", rd, rs1);
                    self.f[rd] = self.round_rm_f64(funct3, false, self.x[rs1] as i32 as i64 as f64);
                    println!(" | rd: {}", self.f[rd]);
                }
                0b1101001 if rs2 == 1 => {
                    print!("fcvt.d.wu x{}, x{}", rd, rs1);
                    self.f[rd] = self.round_rm_f64(funct3, false, self.x[rs1] as u64 as f64);
                    println!(" | rd: {}", self.f[rd]);
                }
                _ => panic!("invalid instruction"),
            },
            _ => panic!("invalid instruction"),
        }
    }

    // Perform rounding based on RM table. (f32)
    #[inline]
    pub fn round_rm_f32(&self, rm: u32, frm: bool, value: f32) -> f32 {
        match rm {
            0b000 => value.round_ties_even(),
            0b001 => value.trunc(),
            0b010 => value.floor(),
            0b011 => value.ceil(),
            0b111 if !frm => self.round_rm_f32((self.csrs[0x003] >> 5) & 0b111, true, value),
            _ => panic!("invalid rounding mode"),
        }
    }

    // Perform rounding based on RM table. (f64)
    #[inline]
    pub fn round_rm_f64(&self, rm: u32, frm: bool, value: f64) -> f64 {
        match rm {
            0b000 => value.round_ties_even(),
            0b001 => value.trunc(),
            0b010 => value.floor(),
            0b011 => value.ceil(),
            0b111 if !frm => self.round_rm_f64((self.csrs[0x003] >> 5) & 0b111, true, value),
            _ => panic!("invalid rounding mode"),
        }
    }

    // Pull a NaN-boxed f32 from f and NaN-unbox it.
    #[inline]
    fn get_f32(&self, reg: usize) -> f32 {
        f32::from_bits((self.f[reg].to_bits() & 0xFFFFFFFF) as u32)
    }

    // NaN-box an f32 and place it in f.
    #[inline]
    fn set_f32(&mut self, reg: usize, value: f32) {
        self.f[reg] = f64::from_bits(0xFFFFFFFF00000000u64 | value.to_bits() as u64);
    }
}
