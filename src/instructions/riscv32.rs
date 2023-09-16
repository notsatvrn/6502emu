use std::fmt::Display;
pub use std::sync::atomic::Ordering;

// RV32I - branches
#[derive(Clone, Copy, Debug)]
pub struct Branch {
    pub rs1: usize,
    pub rs2: usize,
    pub offset: i32,
    pub mode: BranchMode,
}

impl Display for Branch {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} x{}, {}(x{})",
            self.mode, self.rs1, self.offset, self.rs2
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BranchMode {
    Equal,
    NotEqual,
    LessThan,
    GreaterOrEqual,
    LessThanUnsigned,
    GreaterOrEqualUnsigned,
}

impl Display for BranchMode {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                BranchMode::Equal => "beq",
                BranchMode::NotEqual => "bne",
                BranchMode::LessThan => "blt",
                BranchMode::GreaterOrEqual => "bge",
                BranchMode::LessThanUnsigned => "bltu",
                BranchMode::GreaterOrEqualUnsigned => "bgeu",
            }
        )
    }
}

// RV32I - load
#[derive(Clone, Copy, Debug)]
pub struct Load {
    pub rd: usize,
    pub rs1: usize,
    pub offset: i32,
    pub mode: LoadMode,
}

impl Display for Load {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} x{}, {}(x{})",
            self.mode, self.rd, self.offset, self.rs1
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub enum LoadMode {
    Byte,
    HalfWord,
    Word,
    UnsignedByte,
    UnsignedHalfWord,
}

impl Display for LoadMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                LoadMode::Byte => "lb",
                LoadMode::HalfWord => "lh",
                LoadMode::Word => "lw",
                LoadMode::UnsignedByte => "lbu",
                LoadMode::UnsignedHalfWord => "lhu",
            }
        )
    }
}

// RV32I - store
#[derive(Clone, Copy, Debug)]
pub struct Store {
    pub rs1: usize,
    pub rs2: usize,
    pub offset: i32,
    pub mode: StoreMode,
}

impl Display for Store {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} x{}, {}(x{})",
            self.mode, self.rs1, self.offset, self.rs2
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub enum StoreMode {
    Byte,
    HalfWord,
    Word,
}

impl Display for StoreMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                StoreMode::Byte => "sb",
                StoreMode::HalfWord => "sh",
                StoreMode::Word => "sw",
            }
        )
    }
}

// RV32I - immediate integer operation
#[derive(Clone, Copy, Debug)]
pub struct IMMOp {
    pub rd: usize,
    pub rs1: usize,
    pub immediate: i32,
    pub mode: IMMOpMode,
}

impl Display for IMMOp {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} x{}, x{}, {}",
            self.mode, self.rd, self.rs1, self.immediate
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub enum IMMOpMode {
    Add,
    SetLessThan,
    SetLessThanUnsigned,
    ExclusiveOr,
    Or,
    And,
}

impl Display for IMMOpMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                IMMOpMode::Add => "addi",
                IMMOpMode::SetLessThan => "slti",
                IMMOpMode::SetLessThanUnsigned => "sltiu",
                IMMOpMode::ExclusiveOr => "xori",
                IMMOpMode::Or => "ori",
                IMMOpMode::And => "andi",
            }
        )
    }
}

// RV32I - shift mode
#[derive(Clone, Copy, Debug)]
pub enum ShiftMode {
    LogicalLeft,
    LogicalRight,
    ArithmeticRight,
}

impl Display for ShiftMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ShiftMode::LogicalLeft => "slli",
                ShiftMode::LogicalRight => "srli",
                ShiftMode::ArithmeticRight => "srai",
            }
        )
    }
}

// RV32I - immediate integer shift
#[derive(Clone, Copy, Debug)]
pub struct IMMShift {
    pub rd: usize,
    pub rs1: usize,
    pub amount: u32,
    pub mode: ShiftMode,
}

impl Display for IMMShift {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} x{}, x{}, {}",
            self.mode, self.rd, self.rs1, self.amount
        )
    }
}

// RV32I - integer operation
#[derive(Clone, Copy, Debug)]
pub struct IntOp {
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
    pub mode: IntOpMode,
}

impl Display for IntOp {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} x{}, x{}, x{}",
            self.mode, self.rd, self.rs1, self.rs2
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub enum IntOpMode {
    Add,
    Subtract,
    SetLessThan,
    SetLessThanUnsigned,
    ExclusiveOr,
    Or,
    And,
}

impl Display for IntOpMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                IntOpMode::Add => "add",
                IntOpMode::Subtract => "sub",
                IntOpMode::SetLessThan => "slt",
                IntOpMode::SetLessThanUnsigned => "sltu",
                IntOpMode::ExclusiveOr => "xor",
                IntOpMode::Or => "or",
                IntOpMode::And => "and",
            }
        )
    }
}

// RV32I - integer shift
#[derive(Clone, Copy, Debug)]
pub struct IntShift {
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
    pub mode: ShiftMode,
}

impl Display for IntShift {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} x{}, x{}, x{}",
            self.mode, self.rd, self.rs1, self.rs2
        )
    }
}

// Zicsr
#[derive(Clone, Copy, Debug)]
pub struct CSR {
    pub rd: usize,
    pub source: CSRSource,
    pub mode: CSRMode,
    pub csr: usize,
}

impl Display for CSR {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mode_source = if let CSRSource::Immediate(_) = self.source {
            "i"
        } else {
            ""
        };

        write!(
            f,
            "{}{} x{}, {}, {}",
            self.mode, mode_source, self.rd, self.source, self.csr
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub enum CSRSource {
    Register(usize),
    Immediate(u32),
}

impl Display for CSRSource {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CSRSource::Register(register) => write!(f, "x{}", register),
            CSRSource::Immediate(immediate) => write!(f, "{}", immediate),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum CSRMode {
    ReadWrite,
    ReadSetBits,
    ReadClearBits,
}

impl Display for CSRMode {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                CSRMode::ReadWrite => "csrrw",
                CSRMode::ReadSetBits => "csrrs",
                CSRMode::ReadClearBits => "csrrc",
            }
        )
    }
}

// RV32M
#[derive(Clone, Copy, Debug)]
pub struct MulOp {
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
    pub mode: MulOpMode,
}

impl Display for MulOp {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} x{}, x{}, x{}",
            self.mode, self.rd, self.rs1, self.rs2
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub enum MulOpMode {
    Multiply,
    MultiplyHull,
    MultiplyHullSignedUnsigned,
    MultiplyHullUnsigned,
    Divide,
    DivideUnsigned,
    Remainder,
    RemainderUnsigned,
}

impl Display for MulOpMode {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                MulOpMode::Multiply => "mul",
                MulOpMode::MultiplyHull => "mulh",
                MulOpMode::MultiplyHullSignedUnsigned => "mulhsu",
                MulOpMode::MultiplyHullUnsigned => "mulhu",
                MulOpMode::Divide => "div",
                MulOpMode::DivideUnsigned => "divu",
                MulOpMode::Remainder => "rem",
                MulOpMode::RemainderUnsigned => "remu",
            }
        )
    }
}

// RV32A
#[derive(Clone, Copy, Debug)]
pub struct Atomic {
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
    pub ordering: Ordering,
    pub mode: AtomicMode,
}

pub fn fmt_ordering(ordering: Ordering) -> String {
    match ordering {
        Ordering::SeqCst => "1, 1".to_owned(),
        Ordering::Acquire => "1, 0".to_owned(),
        Ordering::Release => "0, 1".to_owned(),
        Ordering::Relaxed => "0, 0".to_owned(),
        _ => panic!("illegal instruction"),
    }
}

impl Display for Atomic {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} x{}, x{}, x{}, {}",
            self.mode,
            self.rd,
            self.rs1,
            self.rs2,
            fmt_ordering(self.ordering)
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub enum AtomicMode {
    LoadReservedWord,
    StoreConditionalWord,
    SwapWord,
    AddWord,
    ExclusiveOrWord,
    AndWord,
    OrWord,
    MinimumWord,
    MaximumWord,
    MinimumUnsignedWord,
    MaximumUnsignedWord,
}

impl Display for AtomicMode {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                AtomicMode::LoadReservedWord => "lr.w",
                AtomicMode::StoreConditionalWord => "sc.w",
                AtomicMode::SwapWord => "amoswap.w",
                AtomicMode::AddWord => "amoadd.w",
                AtomicMode::ExclusiveOrWord => "amoxor.w",
                AtomicMode::AndWord => "amoand.w",
                AtomicMode::OrWord => "amoor.w",
                AtomicMode::MinimumWord => "amomin.w",
                AtomicMode::MaximumWord => "amomax.w",
                AtomicMode::MinimumUnsignedWord => "amominu.w",
                AtomicMode::MaximumUnsignedWord => "amomaxu.w",
            }
        )
    }
}

// RV32F/D - precision
#[derive(Clone, Copy, Debug)]
pub enum FPPrecision {
    Single,
    Double,
}

impl Display for FPPrecision {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                FPPrecision::Single => "s",
                FPPrecision::Double => "d",
            }
        )
    }
}

// RV32F/D - load
#[derive(Clone, Copy, Debug)]
pub struct FPLoad {
    pub rd: usize,
    pub rs1: usize,
    pub offset: i32,
    pub precision: FPPrecision,
}

impl Display for FPLoad {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fl{} x{}, {}(x{})",
            match self.precision {
                FPPrecision::Single => "w",
                FPPrecision::Double => "d",
            },
            self.rd,
            self.offset,
            self.rs1
        )
    }
}

// RV32F/D - store
#[derive(Clone, Copy, Debug)]
pub struct FPStore {
    pub rs1: usize,
    pub rs2: usize,
    pub offset: i32,
    pub precision: FPPrecision,
}

impl Display for FPStore {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fs{} x{}, {}(x{})",
            match self.precision {
                FPPrecision::Single => "w",
                FPPrecision::Double => "d",
            },
            self.rs1,
            self.offset,
            self.rs2
        )
    }
}

// RV32F/D - rounding mode
#[derive(Clone, Copy, Debug)]
pub enum FPRoundingMode {
    NearestTiesEven,
    Zero,
    Down,
    Up,
    NearestMaxMagnitude,
    Dynamic,
}

impl FPRoundingMode {
    #[inline]
    pub fn decode(rm: u32) -> FPRoundingMode {
        match rm {
            0b000 => FPRoundingMode::NearestTiesEven,
            0b001 => FPRoundingMode::Zero,
            0b010 => FPRoundingMode::Down,
            0b011 => FPRoundingMode::Up,
            0b100 => FPRoundingMode::NearestMaxMagnitude,
            0b111 => FPRoundingMode::Dynamic,
            _ => panic!("invalid rounding mode"),
        }
    }

    #[inline]
    pub fn apply_f64(&self, csr_rm: u32, frm: bool, value: f64) -> f64 {
        match self {
            FPRoundingMode::NearestTiesEven => value.round_ties_even(),
            FPRoundingMode::Zero => value.trunc(),
            FPRoundingMode::Down => value.floor(),
            FPRoundingMode::Up => value.ceil(),
            FPRoundingMode::NearestMaxMagnitude => value.round(),
            FPRoundingMode::Dynamic if !frm => {
                FPRoundingMode::decode(csr_rm).apply_f64(0, true, value)
            }
            _ => panic!("invalid rounding mode"),
        }
    }

    #[inline]
    pub fn apply_f32(&self, csr_rm: u32, frm: bool, value: f32) -> f32 {
        match self {
            FPRoundingMode::NearestTiesEven => value.round_ties_even(),
            FPRoundingMode::Zero => value.trunc(),
            FPRoundingMode::Down => value.floor(),
            FPRoundingMode::Up => value.ceil(),
            FPRoundingMode::NearestMaxMagnitude => value.round(),
            FPRoundingMode::Dynamic if !frm => {
                FPRoundingMode::decode(csr_rm).apply_f32(0, true, value)
            }
            _ => panic!("invalid rounding mode"),
        }
    }
}

// RV32F/D - return mode
#[derive(Clone, Copy, Debug)]
pub enum FPReturnMode {
    Double,
    Single,
    Integer,
}

// RV32F/D - fused multiply op
#[derive(Clone, Copy, Debug)]
pub struct FPFusedMultiplyOp {
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
    pub rs3: usize,
    pub add: bool,
    pub positive: bool,
    pub rounding: FPRoundingMode,
    pub precision: FPPrecision,
}

impl Display for FPFusedMultiplyOp {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "f")?;

        if !self.positive {
            write!(f, "n")?;
        }

        write!(f, "m")?;

        if self.add {
            write!(f, "add")?;
        } else {
            write!(f, "sub")?;
        }

        write!(
            f,
            ".{} x{}, x{}, x{}, x{}",
            self.precision, self.rd, self.rs1, self.rs2, self.rs3,
        )
    }
}

// RV32F
#[derive(Clone, Copy, Debug)]
pub struct FPSingleOp {
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
    pub rounding: FPRoundingMode,
    pub mode: FPSingleOpMode,
    pub ret: FPReturnMode,
}

impl Display for FPSingleOp {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum FPSingleOpMode {
    Add,
    Subtract,
    Multiply,
    Divide,
    SquareRoot,
    SignInject,
    SignInjectNot,
    SignInjectExclusiveOr,
    Minimum,
    Maximum,
    Equals,
    LessThan,
    LessThanOrEqual,
    Class,
    ConvertWordFromSingle,
    ConvertUnsignedWordFromSingle,
    ConvertSingleFromWord,
    ConvertSingleFromUnsignedWord,
    MoveWordFromSingle,
    MoveSingleFromWord,
}

impl FPSingleOpMode {
    fn return_mode(&self) -> FPReturnMode {
        match self {
            Self::Equals
            | Self::LessThan
            | Self::LessThanOrEqual
            | Self::Class
            | Self::ConvertWordFromSingle
            | Self::ConvertUnsignedWordFromSingle
            | Self::MoveWordFromSingle => FPReturnMode::Integer,
            _ => FPReturnMode::Single,
        }
    }
}

impl Display for FPSingleOpMode {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}

// RV32D
#[derive(Clone, Copy, Debug)]
pub struct FPDoubleOp {
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
    pub rounding: FPRoundingMode,
    pub mode: FPDoubleOpMode,
    pub ret: FPReturnMode,
}

impl Display for FPDoubleOp {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum FPDoubleOpMode {
    Add,
    Subtract,
    Multiply,
    Divide,
    SquareRoot,
    SignInject,
    SignInjectNot,
    SignInjectExclusiveOr,
    Minimum,
    Maximum,
    Equals,
    LessThan,
    LessThanOrEqual,
    Class,
    ConvertSingleFromDouble,
    ConvertDoubleFromSingle,
    ConvertWordFromDouble,
    ConvertUnsignedWordFromDouble,
    ConvertDoubleFromWord,
    ConvertDoubleFromUnsignedWord,
}

impl FPDoubleOpMode {
    fn return_mode(&self) -> FPReturnMode {
        match self {
            Self::Equals
            | Self::LessThan
            | Self::LessThanOrEqual
            | Self::Class
            | Self::ConvertWordFromDouble
            | Self::ConvertUnsignedWordFromDouble => FPReturnMode::Integer,
            Self::ConvertSingleFromDouble => FPReturnMode::Single,
            _ => FPReturnMode::Double,
        }
    }
}

impl Display for FPDoubleOpMode {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}

// RV32 instruction
#[derive(Clone, Copy, Debug)]
pub enum Instruction {
    LUI(usize, u32),                    // rd, immediate
    AUIPC(usize, u32),                  // rd, immediate
    JAL(usize, i32),                    // rd, offset
    JALR(usize, i32, usize),            // rd, offset(rs1)
    Branch(Branch),                     // rs1, offset(rs2)
    Load(Load),                         // rd, offset(rs1)
    Store(Store),                       // rs1, offset(rs2)
    IMMOp(IMMOp),                       // rd, rs1, imm
    IMMShift(IMMShift),                 // rd, rs1, imm
    IntOp(IntOp),                       // rd, rs1, rs2
    IntShift(IntShift),                 // rd, rs1, rs2
    Fence(usize, usize, u32, u32, u32), // rd, rs1, fm, pred, succ
    ECall,
    EBreak,

    FenceI(usize, usize, u32), // rd, rs1, imm
    CSR(CSR),
    MulOp(MulOp),   // rd, rs1, rs2
    Atomic(Atomic), // rd, rs1, rs2, aq, rl

    FPLoad(FPLoad),
    FPStore(FPStore),
    FPFusedMultiplyOp(FPFusedMultiplyOp),
    FPSingleOp(FPSingleOp),
    FPDoubleOp(FPDoubleOp),
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::LUI(rd, immediate) => write!(f, "lui x{}, {}", rd, immediate),
            Instruction::AUIPC(rd, immediate) => write!(f, "auipc x{}, {}", rd, immediate),
            Instruction::JAL(rd, offset) => write!(f, "jal x{}, {}", rd, offset),
            Instruction::JALR(rd, offset, rs1) => write!(f, "jalr x{}, {}(x{})", rd, offset, rs1),
            Instruction::Branch(branch) => write!(f, "{}", branch),
            Instruction::Load(load) => write!(f, "{}", load),
            Instruction::Store(store) => write!(f, "{}", store),
            Instruction::IMMOp(op) => write!(f, "{}", op),
            Instruction::IMMShift(shift) => write!(f, "{}", shift),
            Instruction::IntOp(op) => write!(f, "{}", op),
            Instruction::IntShift(shift) => write!(f, "{}", shift),
            Instruction::Fence(rd, rs1, fm, pred, succ) => {
                write!(f, "fence x{}, x{}, {}, {}, {}", rd, rs1, fm, pred, succ)
            }
            Instruction::ECall => write!(f, "ecall"),
            Instruction::EBreak => write!(f, "ebreak"),
            // Zifencei
            Instruction::FenceI(rd, rs1, immediate) => {
                write!(f, "fencei x{}, x{}, {}", rd, rs1, immediate)
            }
            Instruction::CSR(csr) => write!(f, "{}", csr),
            Instruction::MulOp(op) => write!(f, "{}", op),
            Instruction::Atomic(atomic) => write!(f, "{}", atomic),
            Instruction::FPLoad(load) => write!(f, "{}", load),
            Instruction::FPStore(store) => write!(f, "{}", store),
            Instruction::FPFusedMultiplyOp(op) => write!(f, "{}", op),
            Instruction::FPSingleOp(op) => write!(f, "{}", op),
            Instruction::FPDoubleOp(op) => write!(f, "{}", op),
        }
    }
}

pub enum CompressedInstruction {
    NOP,
}

// Perform sign extension.
#[inline]
pub fn sign_extend(n: u32, b: u8) -> i32 {
    let shift = 32 - b;
    (n << shift) as i32 >> shift
}

// Decode a compressed RISC-V / RV32C instruction.
pub fn decode_compressed(inst: u16) -> CompressedInstruction {
    CompressedInstruction::NOP
}

// Decode a full RISC-V / RV32 ISA instruction.
pub fn decode_full(inst: u32) -> Instruction {
    let opcode = inst & 0b1111111;
    let rd = ((inst >> 7) & 0b11111) as usize;
    let funct3 = (inst >> 12) & 0b111;
    let rs1 = ((inst >> 15) & 0b11111) as usize;
    let rs2 = ((inst >> 20) & 0b11111) as usize;
    let funct7 = inst >> 25;
    let funct2 = funct7 & 0b11;
    let rs3 = (inst >> 27) as usize;

    let int_op = funct7 == 0 || (funct3 == 0b101 && funct7 == 0b0100000);
    let funct3_01 = funct3 & 0b11 == 0b01;

    match opcode {
        // RV32I
        0b0110111 => Instruction::LUI(rd, inst >> 12),
        0b0010111 => Instruction::AUIPC(rd, inst >> 12),
        0b1101111 => {
            let umm = ((inst >> 31) << 19)
                | (((inst >> 12) & 0b11111111) << 11)
                | (((inst >> 20) & 0b1) << 10)
                | ((inst >> 21) & 0b1111111111);
            Instruction::JAL(rd, sign_extend(umm, 20))
        }
        0b1100111 => Instruction::JALR(rd, sign_extend(inst >> 20, 12), rs1),
        0b1100011 => {
            let umm = ((inst >> 31) << 11)
                | (((inst >> 7) & 0b1) << 10)
                | (((inst >> 25) & 0b111111) << 4)
                | ((inst >> 8) & 0b1111);

            let mode = match funct3 {
                0b000 => BranchMode::Equal,
                0b001 => BranchMode::NotEqual,
                0b100 => BranchMode::LessThan,
                0b101 => BranchMode::GreaterOrEqual,
                0b110 => BranchMode::LessThanUnsigned,
                0b111 => BranchMode::GreaterOrEqualUnsigned,
                _ => panic!("illegal instruction"),
            };

            Instruction::Branch(Branch {
                rs1,
                rs2,
                offset: (sign_extend(umm, 12) * 2) - 4,
                mode,
            })
        }
        0b0000011 => {
            let offset = sign_extend(inst >> 20, 12);

            let mode = match funct3 {
                0b000 => LoadMode::Byte,
                0b001 => LoadMode::HalfWord,
                0b010 => LoadMode::Word,
                0b100 => LoadMode::UnsignedByte,
                0b101 => LoadMode::UnsignedHalfWord,
                _ => panic!("illegal instruction"),
            };

            Instruction::Load(Load {
                rd,
                rs1,
                offset,
                mode,
            })
        }
        0b0100011 => {
            let offset = sign_extend(((inst >> 7) & 0b11111) | ((inst >> 25) << 5), 12);

            let mode = match funct3 {
                0b000 => StoreMode::Byte,
                0b001 => StoreMode::HalfWord,
                0b010 => StoreMode::Word,
                _ => panic!("illegal instruction"),
            };

            Instruction::Store(Store {
                rs1,
                rs2,
                offset,
                mode,
            })
        }
        0b0010011 if !funct3_01 => {
            let immediate = sign_extend(inst >> 20, 12);

            let mode = match funct3 {
                0b000 => IMMOpMode::Add,
                0b010 => IMMOpMode::SetLessThan,
                0b011 => IMMOpMode::SetLessThanUnsigned,
                0b100 => IMMOpMode::ExclusiveOr,
                0b110 => IMMOpMode::Or,
                0b111 => IMMOpMode::And,
                _ => panic!("illegal instruction"),
            };

            Instruction::IMMOp(IMMOp {
                rd,
                rs1,
                immediate,
                mode,
            })
        }
        0b0010011 => {
            let amount = (inst >> 20) & 0b11111;

            let mode = if funct3 == 0b001 {
                ShiftMode::LogicalLeft
            } else if funct7 == 0 {
                ShiftMode::LogicalRight
            } else if funct7 == 0b0100000 {
                ShiftMode::ArithmeticRight
            } else {
                panic!("illegal instruction");
            };

            Instruction::IMMShift(IMMShift {
                rd,
                rs1,
                amount,
                mode,
            })
        }
        0b0110011 if !funct3_01 && int_op => {
            let mode = match funct3 {
                0b000 => match funct7 {
                    0b0000000 => IntOpMode::Add,
                    0b0100000 => IntOpMode::Subtract,
                    _ => panic!("illegal instruction"),
                },
                0b010 => IntOpMode::SetLessThan,
                0b011 => IntOpMode::SetLessThanUnsigned,
                0b100 => IntOpMode::ExclusiveOr,
                0b110 => IntOpMode::Or,
                0b111 => IntOpMode::And,
                _ => panic!("illegal instruction"),
            };

            Instruction::IntOp(IntOp { rd, rs1, rs2, mode })
        }
        0b0110011 if funct3_01 && int_op => {
            let mode = if funct3 == 1 {
                ShiftMode::LogicalLeft
            } else if funct7 == 0 {
                ShiftMode::LogicalRight
            } else if funct7 == 0b0100000 {
                ShiftMode::ArithmeticRight
            } else {
                panic!("illegal instruction");
            };

            Instruction::IntShift(IntShift { rd, rs1, rs2, mode })
        }
        0b0001111 => match funct3 {
            0b000 => Instruction::Fence(
                rd,
                rs1,
                inst >> 28,
                (inst >> 24) & 0b1111,
                (inst >> 20) & 0b1111,
            ),
            // Zifencei
            0b001 => Instruction::FenceI(rd, rs1, inst >> 20),
            _ => panic!("illegal instruction"),
        },
        0b1110011 => {
            let imm = inst >> 20;
            let middle = (inst << 12) >> 19;

            if middle == 0 {
                match imm {
                    0 => Instruction::ECall,
                    1 => Instruction::EBreak,
                    _ => panic!("illegal instruction"),
                }
            } else {
                // Zicsr
                let source = if (funct3 >> 2) == 0 {
                    CSRSource::Register(rs1)
                } else {
                    CSRSource::Immediate(rs1 as u32)
                };

                let funct3_2 = funct3 & 0b11;
                let mode = if funct3_2 == 0b01 {
                    CSRMode::ReadWrite
                } else if funct3_2 == 0b10 {
                    CSRMode::ReadSetBits
                } else {
                    CSRMode::ReadClearBits
                };

                Instruction::CSR(CSR {
                    rd,
                    source,
                    mode,
                    csr: imm as usize,
                })
            }
        }
        // RV32M
        0b0110011 if funct3_01 && funct7 == 1 => {
            let mode = match funct3 {
                0b000 => MulOpMode::Multiply,
                0b001 => MulOpMode::MultiplyHull,
                0b010 => MulOpMode::MultiplyHullSignedUnsigned,
                0b011 => MulOpMode::MultiplyHullUnsigned,
                0b100 => MulOpMode::Divide,
                0b101 => MulOpMode::DivideUnsigned,
                0b110 => MulOpMode::Remainder,
                0b111 => MulOpMode::RemainderUnsigned,
                _ => panic!("illegal instruction"),
            };

            Instruction::MulOp(MulOp { rd, rs1, rs2, mode })
        }
        // RV32A
        0b0101111 => {
            if funct3 != 0b010 {
                panic!("illegal instruction: no RV64A on RV32");
            }

            let acquire = (funct2 & 0b10) != 0;
            let release = (funct2 & 0b01) != 0;

            let ordering = if acquire {
                if release {
                    Ordering::SeqCst
                } else {
                    Ordering::Acquire
                }
            } else if release {
                Ordering::Release
            } else {
                Ordering::Relaxed
            };

            let mode = match rs3 {
                0b00010 if rs2 == 0b00000 => AtomicMode::LoadReservedWord,
                0b00011 => AtomicMode::StoreConditionalWord,
                0b00001 => AtomicMode::SwapWord,
                0b00000 => AtomicMode::AddWord,
                0b00100 => AtomicMode::ExclusiveOrWord,
                0b01100 => AtomicMode::AndWord,
                0b01000 => AtomicMode::OrWord,
                0b10000 => AtomicMode::MinimumWord,
                0b10100 => AtomicMode::MaximumWord,
                0b11000 => AtomicMode::MinimumUnsignedWord,
                0b11100 => AtomicMode::MaximumUnsignedWord,
                _ => panic!("illegal instruction"),
            };

            Instruction::Atomic(Atomic {
                rd,
                rs1,
                rs2,
                ordering,
                mode,
            })
        }
        // RV32F/D
        0b0000111 if (funct3 >> 1) == 1 => {
            let offset = sign_extend(inst >> 20, 12);

            let precision = if funct3 == 0b010 {
                FPPrecision::Single
            } else {
                FPPrecision::Double
            };

            Instruction::FPLoad(FPLoad {
                rd,
                rs1,
                offset,
                precision,
            })
        }
        0b0100111 if (funct3 >> 1) == 1 => {
            let offset = sign_extend(((inst >> 7) & 0b11111) | ((inst >> 25) << 5), 12);

            let precision = if funct3 == 0b010 {
                FPPrecision::Single
            } else {
                FPPrecision::Double
            };

            Instruction::FPStore(FPStore {
                rs1,
                rs2,
                offset,
                precision,
            })
        }
        0b1000011 | 0b1000111 | 0b1001011 | 0b1001111 if funct2 & 0b10 == 0 => {
            let precision = if funct2 & 1 == 0 {
                FPPrecision::Single
            } else {
                FPPrecision::Double
            };

            Instruction::FPFusedMultiplyOp(FPFusedMultiplyOp {
                rd,
                rs1,
                rs2,
                rs3,
                add: opcode == 0b1000011 || opcode == 0b1001111,
                positive: opcode == 0b1000011 || opcode == 0b1000111,
                precision,
                rounding: FPRoundingMode::decode(funct3),
            })
        }
        // RV32F
        0b1010011 if funct7 != 0b0100000 && funct2 == 0b00 => {
            let mode = match funct7 {
                0b0000000 => FPSingleOpMode::Add,
                0b0000100 => FPSingleOpMode::Subtract,
                0b0001000 => FPSingleOpMode::Multiply,
                0b0001100 => FPSingleOpMode::Divide,
                0b0101100 if rs2 == 0 => FPSingleOpMode::SquareRoot,
                0b0010000 => match funct3 {
                    0 => FPSingleOpMode::SignInject,
                    1 => FPSingleOpMode::SignInjectNot,
                    2 => FPSingleOpMode::SignInjectExclusiveOr,
                    _ => panic!("illegal instruction"),
                },
                0b0010100 if (funct3 >> 1) == 0 => {
                    if funct3 == 0 {
                        FPSingleOpMode::Minimum
                    } else {
                        FPSingleOpMode::Maximum
                    }
                }
                0b1010000 => match funct3 {
                    2 => FPSingleOpMode::Equals,
                    1 => FPSingleOpMode::LessThan,
                    0 => FPSingleOpMode::LessThanOrEqual,
                    _ => panic!("illegal instruction"),
                },
                0b1110000 if rs2 == 0 && (funct3 >> 1) == 0 => {
                    if funct3 == 0 {
                        FPSingleOpMode::MoveWordFromSingle
                    } else {
                        FPSingleOpMode::Class
                    }
                }
                0b1100001 if (rs2 >> 1) == 0 => {
                    if funct3 == 0 {
                        FPSingleOpMode::ConvertWordFromSingle
                    } else {
                        FPSingleOpMode::ConvertUnsignedWordFromSingle
                    }
                }
                0b1101000 if (rs2 >> 1) == 0 => {
                    if funct3 == 0 {
                        FPSingleOpMode::ConvertSingleFromWord
                    } else {
                        FPSingleOpMode::ConvertSingleFromUnsignedWord
                    }
                }
                0b1111000 if rs2 == 0 && funct3 == 0 => FPSingleOpMode::MoveSingleFromWord,
                _ => panic!("illegal instruction"),
            };

            Instruction::FPSingleOp(FPSingleOp {
                rd,
                rs1,
                rs2,
                mode,
                rounding: FPRoundingMode::decode(funct3),
                ret: mode.return_mode(),
            })
        }
        // RV32D
        0b1010011 if funct7 == 0b0100000 || funct2 == 0b01 => {
            let mode = match funct7 {
                0b0000001 => FPDoubleOpMode::Add,
                0b0000101 => FPDoubleOpMode::Subtract,
                0b0001001 => FPDoubleOpMode::Multiply,
                0b0001101 => FPDoubleOpMode::Divide,
                0b0101101 if rs2 == 0 => FPDoubleOpMode::SquareRoot,
                0b0010001 => match funct3 {
                    0 => FPDoubleOpMode::SignInject,
                    1 => FPDoubleOpMode::SignInjectNot,
                    2 => FPDoubleOpMode::SignInjectExclusiveOr,
                    _ => panic!("illegal instruction"),
                },
                0b0010101 if (funct3 >> 1) == 0 => {
                    if funct3 == 0 {
                        FPDoubleOpMode::Minimum
                    } else {
                        FPDoubleOpMode::Maximum
                    }
                }
                0b1010001 => match funct3 {
                    2 => FPDoubleOpMode::Equals,
                    1 => FPDoubleOpMode::LessThan,
                    0 => FPDoubleOpMode::LessThanOrEqual,
                    _ => panic!("illegal instruction"),
                },
                0b0100000 if rs2 == 1 => FPDoubleOpMode::ConvertSingleFromDouble,
                0b0100001 if rs2 == 0 => FPDoubleOpMode::ConvertDoubleFromSingle,
                0b1110001 if rs2 == 0 && funct3 == 1 => FPDoubleOpMode::Class,
                0b1100001 if (rs2 >> 1) == 0 => {
                    if funct3 == 0 {
                        FPDoubleOpMode::ConvertWordFromDouble
                    } else {
                        FPDoubleOpMode::ConvertUnsignedWordFromDouble
                    }
                }
                0b1101001 if (rs2 >> 1) == 0 => {
                    if funct3 == 0 {
                        FPDoubleOpMode::ConvertDoubleFromWord
                    } else {
                        FPDoubleOpMode::ConvertDoubleFromUnsignedWord
                    }
                }
                _ => panic!("illegal instruction"),
            };

            Instruction::FPDoubleOp(FPDoubleOp {
                rd,
                rs1,
                rs2,
                mode,
                rounding: FPRoundingMode::decode(funct3),
                ret: mode.return_mode(),
            })
        }
        _ => panic!("illegal instruction"),
    }
}

// Encode a full RISC-V / RV32 ISA instruction as a u32.
