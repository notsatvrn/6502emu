use std::fmt::Display;
pub use std::sync::atomic::Ordering;

// RV32I - branches
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
        match self {
            BranchMode::Equal => write!(f, "beq"),
            BranchMode::NotEqual => write!(f, "bne"),
            BranchMode::LessThan => write!(f, "blt"),
            BranchMode::GreaterOrEqual => write!(f, "bge"),
            BranchMode::LessThanUnsigned => write!(f, "bltu"),
            BranchMode::GreaterOrEqualUnsigned => write!(f, "bgeu"),
        }
    }
}

// RV32I - load
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum LoadMode {
    Byte,
    HalfWord,
    Word,
    UnsignedByte,
    UnsignedHalfWord,
}

impl LoadMode {
    pub fn size(&self) -> usize {
        match self {
            Self::Byte | Self::UnsignedByte => 1,
            Self::HalfWord | Self::UnsignedHalfWord => 2,
            Self::Word => 4,
        }
    }
}

impl Display for LoadMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadMode::Byte => write!(f, "lb"),
            LoadMode::HalfWord => write!(f, "lh"),
            LoadMode::Word => write!(f, "lw"),
            LoadMode::UnsignedByte => write!(f, "lbu"),
            LoadMode::UnsignedHalfWord => write!(f, "lhu"),
        }
    }
}

// RV32I - store
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum StoreMode {
    Byte,
    HalfWord,
    Word,
}

impl StoreMode {
    pub fn size(&self) -> usize {
        match self {
            Self::Byte => 1,
            Self::HalfWord => 2,
            Self::Word => 4,
        }
    }
}

impl Display for StoreMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreMode::Byte => write!(f, "sb"),
            StoreMode::HalfWord => write!(f, "sh"),
            StoreMode::Word => write!(f, "sw"),
        }
    }
}

// RV32I - immediate integer operation
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
        match self {
            IMMOpMode::Add => write!(f, "addi"),
            IMMOpMode::SetLessThan => write!(f, "slti"),
            IMMOpMode::SetLessThanUnsigned => write!(f, "sltiu"),
            IMMOpMode::ExclusiveOr => write!(f, "xori"),
            IMMOpMode::Or => write!(f, "ori"),
            IMMOpMode::And => write!(f, "andi"),
        }
    }
}

// RV32I - shift mode
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ShiftMode {
    LogicalLeft,
    LogicalRight,
    ArithmeticRight,
}

impl Display for ShiftMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShiftMode::LogicalLeft => write!(f, "slli"),
            ShiftMode::LogicalRight => write!(f, "srli"),
            ShiftMode::ArithmeticRight => write!(f, "srai"),
        }
    }
}

// RV32I - immediate integer shift
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct IMMShift {
    pub rd: usize,
    pub rs1: usize,
    pub amount: u32,
    pub mode: ShiftMode,
    pub compressed: bool,
}

impl Display for IMMShift {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.compressed {
            write!(f, "c.{} x{}, {}", self.mode, self.rd, self.amount)
        } else {
            write!(
                f,
                "{} x{}, x{}, {}",
                self.mode, self.rd, self.rs1, self.amount
            )
        }
    }
}

// RV32I - integer operation
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
        match self {
            IntOpMode::Add => write!(f, "add"),
            IntOpMode::Subtract => write!(f, "sub"),
            IntOpMode::SetLessThan => write!(f, "slt"),
            IntOpMode::SetLessThanUnsigned => write!(f, "sltu"),
            IntOpMode::ExclusiveOr => write!(f, "xor"),
            IntOpMode::Or => write!(f, "or"),
            IntOpMode::And => write!(f, "and"),
        }
    }
}

// RV32I - integer shift
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct CSR {
    pub rd: usize,
    pub source: CSRSource,
    pub mode: CSRMode,
    pub csr: usize,
}

impl Display for CSR {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.mode)?;

        if let CSRSource::Immediate(_) = self.source {
            write!(f, "i")?;
        }

        write!(f, " x{}, {}, {}", self.rd, self.source, self.csr)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum CSRMode {
    ReadWrite,
    ReadSetBits,
    ReadClearBits,
}

impl Display for CSRMode {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CSRMode::ReadWrite => write!(f, "csrrw"),
            CSRMode::ReadSetBits => write!(f, "csrrs"),
            CSRMode::ReadClearBits => write!(f, "csrrc"),
        }
    }
}

// RV32M
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
        match self {
            MulOpMode::Multiply => write!(f, "mul"),
            MulOpMode::MultiplyHull => write!(f, "mulh"),
            MulOpMode::MultiplyHullSignedUnsigned => write!(f, "mulhsu"),
            MulOpMode::MultiplyHullUnsigned => write!(f, "mulhu"),
            MulOpMode::Divide => write!(f, "div"),
            MulOpMode::DivideUnsigned => write!(f, "divu"),
            MulOpMode::Remainder => write!(f, "rem"),
            MulOpMode::RemainderUnsigned => write!(f, "remu"),
        }
    }
}

// RV32A
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Atomic {
    pub rd: usize,
    pub rs1: usize,
    pub rs2: usize,
    pub ordering: Ordering,
    pub mode: AtomicMode,
}

pub fn fmt_ordering(ordering: Ordering) -> String {
    match ordering {
        Ordering::SeqCst => ".aq.rl".to_owned(),
        Ordering::Acquire => ".aq".to_owned(),
        Ordering::Release => ".rl".to_owned(),
        Ordering::Relaxed => "".to_owned(),
        _ => panic!("illegal instruction"),
    }
}

impl Display for Atomic {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{} x{}, x{}, x{}",
            fmt_ordering(self.ordering),
            self.mode,
            self.rd,
            self.rs1,
            self.rs2,
        )
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
        match self {
            AtomicMode::LoadReservedWord => write!(f, "lr.w"),
            AtomicMode::StoreConditionalWord => write!(f, "sc.w"),
            AtomicMode::SwapWord => write!(f, "amoswap.w"),
            AtomicMode::AddWord => write!(f, "amoadd.w"),
            AtomicMode::ExclusiveOrWord => write!(f, "amoxor.w"),
            AtomicMode::AndWord => write!(f, "amoand.w"),
            AtomicMode::OrWord => write!(f, "amoor.w"),
            AtomicMode::MinimumWord => write!(f, "amomin.w"),
            AtomicMode::MaximumWord => write!(f, "amomax.w"),
            AtomicMode::MinimumUnsignedWord => write!(f, "amominu.w"),
            AtomicMode::MaximumUnsignedWord => write!(f, "amomaxu.w"),
        }
    }
}

// RV32F/D - precision
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum FPPrecision {
    Single,
    Double,
}

impl FPPrecision {
    pub fn size(&self) -> usize {
        match self {
            Self::Single => 4,
            Self::Double => 8,
        }
    }
}

impl Display for FPPrecision {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FPPrecision::Single => write!(f, "s"),
            FPPrecision::Double => write!(f, "d"),
        }
    }
}

// RV32F/D - load
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
            _ => panic!("illegal rounding mode"),
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
            _ => panic!("illegal rounding mode"),
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
            _ => panic!("illegal rounding mode"),
        }
    }
}

// RV32F/D - return mode
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum FPReturnMode {
    Double,
    Single,
    Integer,
}

// RV32F/D - fused multiply op
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
        let res = write!(f, "{} x{}, x{}", self.mode, self.rd, self.rs1);

        if self.mode.uses_rs2() {
            res?;
            write!(f, ", x{}", self.rs2)
        } else {
            res
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
    pub fn return_mode(&self) -> FPReturnMode {
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

    pub fn uses_rs2(&self) -> bool {
        !matches!(
            self,
            Self::SquareRoot
                | Self::ConvertWordFromSingle
                | Self::ConvertUnsignedWordFromSingle
                | Self::MoveWordFromSingle
                | Self::Class
                | Self::ConvertSingleFromUnsignedWord
                | Self::ConvertSingleFromWord
                | Self::MoveSingleFromWord
        )
    }
}

impl Display for FPSingleOpMode {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => write!(f, "fadd.d"),
            Self::Subtract => write!(f, "fsub.d"),
            Self::Multiply => write!(f, "fmul.d"),
            Self::Divide => write!(f, "fdiv.d"),
            Self::SquareRoot => write!(f, "fsqrt.d"),
            Self::SignInject => write!(f, "fsgnj.d"),
            Self::SignInjectNot => write!(f, "fsgnjn.d"),
            Self::SignInjectExclusiveOr => write!(f, "fsgnjx.d"),
            Self::Minimum => write!(f, "fmin.d"),
            Self::Maximum => write!(f, "fmax.d"),
            Self::Equals => write!(f, "feq.d"),
            Self::LessThan => write!(f, "flt.d"),
            Self::LessThanOrEqual => write!(f, "fle.d"),
            Self::Class => write!(f, "fclass.d"),
            Self::ConvertWordFromSingle => write!(f, "fcvt.w.s"),
            Self::ConvertUnsignedWordFromSingle => write!(f, "fcvt.wu.s"),
            Self::ConvertSingleFromWord => write!(f, "fcvt.s.w"),
            Self::ConvertSingleFromUnsignedWord => write!(f, "fcvt.s.wu"),
            Self::MoveWordFromSingle => write!(f, "fmv.x.w"),
            Self::MoveSingleFromWord => write!(f, "fmv.w.x"),
        }
    }
}

// RV32D
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
        let res = write!(f, "{} x{}, x{}", self.mode, self.rd, self.rs1);

        if self.mode.uses_rs2() {
            res?;
            write!(f, ", x{}", self.rs2)
        } else {
            res
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
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

    pub fn uses_rs2(&self) -> bool {
        !matches!(
            self,
            Self::SquareRoot
                | Self::ConvertSingleFromDouble
                | Self::ConvertDoubleFromSingle
                | Self::Class
                | Self::ConvertWordFromDouble
                | Self::ConvertUnsignedWordFromDouble
                | Self::ConvertDoubleFromWord
                | Self::ConvertDoubleFromUnsignedWord
        )
    }
}

impl Display for FPDoubleOpMode {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => write!(f, "fadd.d"),
            Self::Subtract => write!(f, "fsub.d"),
            Self::Multiply => write!(f, "fmul.d"),
            Self::Divide => write!(f, "fdiv.d"),
            Self::SquareRoot => write!(f, "fsqrt.d"),
            Self::SignInject => write!(f, "fsgnj.d"),
            Self::SignInjectNot => write!(f, "fsgnjn.d"),
            Self::SignInjectExclusiveOr => write!(f, "fsgnjx.d"),
            Self::Minimum => write!(f, "fmin.d"),
            Self::Maximum => write!(f, "fmax.d"),
            Self::Equals => write!(f, "feq.d"),
            Self::LessThan => write!(f, "flt.d"),
            Self::LessThanOrEqual => write!(f, "fle.d"),
            Self::Class => write!(f, "fclass.d"),
            Self::ConvertSingleFromDouble => write!(f, "fcvt.s.d"),
            Self::ConvertDoubleFromSingle => write!(f, "fcvt.d.s"),
            Self::ConvertWordFromDouble => write!(f, "fcvt.wu.d"),
            Self::ConvertUnsignedWordFromDouble => write!(f, "fcvt.w.d"),
            Self::ConvertDoubleFromWord => write!(f, "fcvt.d.w"),
            Self::ConvertDoubleFromUnsignedWord => write!(f, "fcvt.d.wu"),
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Instruction {
    Full(FullInstruction),
    Compressed(CompressedInstruction),
}

// A full RISC-V / RV32 instruction.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum FullInstruction {
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

// Perform sign extension on a u32.
#[inline]
pub fn sign_extend_u32(n: u32, b: u8) -> i32 {
    let shift = 32 - b;
    (n << shift) as i32 >> shift
}

impl FullInstruction {
    // Decode a full RISC-V / RV32 instruction.
    pub fn decode(inst: u32) -> FullInstruction {
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
            0b0110111 => FullInstruction::LUI(rd, inst >> 12),
            0b0010111 => FullInstruction::AUIPC(rd, inst >> 12),
            0b1101111 => {
                let umm = ((inst >> 31) << 19)
                    | (((inst >> 12) & 0b11111111) << 11)
                    | (((inst >> 20) & 0b1) << 10)
                    | ((inst >> 21) & 0b1111111111);
                FullInstruction::JAL(rd, sign_extend_u32(umm, 20))
            }
            0b1100111 => FullInstruction::JALR(rd, sign_extend_u32(inst >> 20, 12), rs1),
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

                FullInstruction::Branch(Branch {
                    rs1,
                    rs2,
                    offset: (sign_extend_u32(umm, 12) * 2) - 4,
                    mode,
                })
            }
            0b0000011 => {
                let offset = sign_extend_u32(inst >> 20, 12);

                let mode = match funct3 {
                    0b000 => LoadMode::Byte,
                    0b001 => LoadMode::HalfWord,
                    0b010 => LoadMode::Word,
                    0b100 => LoadMode::UnsignedByte,
                    0b101 => LoadMode::UnsignedHalfWord,
                    _ => panic!("illegal instruction"),
                };

                FullInstruction::Load(Load {
                    rd,
                    rs1,
                    offset,
                    mode,
                })
            }
            0b0100011 => {
                let offset = sign_extend_u32(((inst >> 7) & 0b11111) | ((inst >> 25) << 5), 12);

                let mode = match funct3 {
                    0b000 => StoreMode::Byte,
                    0b001 => StoreMode::HalfWord,
                    0b010 => StoreMode::Word,
                    _ => panic!("illegal instruction"),
                };

                FullInstruction::Store(Store {
                    rs1,
                    rs2,
                    offset,
                    mode,
                })
            }
            0b0010011 if !funct3_01 => {
                let immediate = sign_extend_u32(inst >> 20, 12);

                let mode = match funct3 {
                    0b000 => IMMOpMode::Add,
                    0b010 => IMMOpMode::SetLessThan,
                    0b011 => IMMOpMode::SetLessThanUnsigned,
                    0b100 => IMMOpMode::ExclusiveOr,
                    0b110 => IMMOpMode::Or,
                    0b111 => IMMOpMode::And,
                    _ => panic!("illegal instruction"),
                };

                FullInstruction::IMMOp(IMMOp {
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

                FullInstruction::IMMShift(IMMShift {
                    rd,
                    rs1,
                    amount,
                    mode,
                    compressed: false,
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

                FullInstruction::IntOp(IntOp { rd, rs1, rs2, mode })
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

                FullInstruction::IntShift(IntShift { rd, rs1, rs2, mode })
            }
            0b0001111 => match funct3 {
                0b000 => FullInstruction::Fence(
                    rd,
                    rs1,
                    inst >> 28,
                    (inst >> 24) & 0b1111,
                    (inst >> 20) & 0b1111,
                ),
                // Zifencei
                0b001 => FullInstruction::FenceI(rd, rs1, inst >> 20),
                _ => panic!("illegal instruction"),
            },
            0b1110011 => {
                let imm = inst >> 20;
                let middle = (inst << 12) >> 19;

                if middle == 0 {
                    match imm {
                        0 => FullInstruction::ECall,
                        1 => FullInstruction::EBreak,
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

                    FullInstruction::CSR(CSR {
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

                FullInstruction::MulOp(MulOp { rd, rs1, rs2, mode })
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

                FullInstruction::Atomic(Atomic {
                    rd,
                    rs1,
                    rs2,
                    ordering,
                    mode,
                })
            }
            // RV32F/D
            0b0000111 if (funct3 >> 1) == 1 => {
                let offset = sign_extend_u32(inst >> 20, 12);

                let precision = if funct3 == 0b010 {
                    FPPrecision::Single
                } else {
                    FPPrecision::Double
                };

                FullInstruction::FPLoad(FPLoad {
                    rd,
                    rs1,
                    offset,
                    precision,
                })
            }
            0b0100111 if (funct3 >> 1) == 1 => {
                let offset = sign_extend_u32(((inst >> 7) & 0b11111) | ((inst >> 25) << 5), 12);

                let precision = if funct3 == 0b010 {
                    FPPrecision::Single
                } else {
                    FPPrecision::Double
                };

                FullInstruction::FPStore(FPStore {
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

                FullInstruction::FPFusedMultiplyOp(FPFusedMultiplyOp {
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

                FullInstruction::FPSingleOp(FPSingleOp {
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

                FullInstruction::FPDoubleOp(FPDoubleOp {
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
}

impl Display for FullInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FullInstruction::LUI(rd, immediate) => write!(f, "lui x{}, {}", rd, immediate),
            FullInstruction::AUIPC(rd, immediate) => write!(f, "auipc x{}, {}", rd, immediate),
            FullInstruction::JAL(rd, offset) => write!(f, "jal x{}, {}", rd, offset),
            FullInstruction::JALR(rd, offset, rs1) => {
                write!(f, "jalr x{}, {}(x{})", rd, offset, rs1)
            }
            FullInstruction::Branch(branch) => write!(f, "{}", branch),
            FullInstruction::Load(load) => write!(f, "{}", load),
            FullInstruction::Store(store) => write!(f, "{}", store),
            FullInstruction::IMMOp(op) => write!(f, "{}", op),
            FullInstruction::IMMShift(shift) => write!(f, "{}", shift),
            FullInstruction::IntOp(op) => write!(f, "{}", op),
            FullInstruction::IntShift(shift) => write!(f, "{}", shift),
            FullInstruction::Fence(rd, rs1, fm, pred, succ) => {
                write!(f, "fence x{}, x{}, {}, {}, {}", rd, rs1, fm, pred, succ)
            }
            FullInstruction::ECall => write!(f, "ecall"),
            FullInstruction::EBreak => write!(f, "ebreak"),
            // Zifencei
            FullInstruction::FenceI(rd, rs1, immediate) => {
                write!(f, "fencei x{}, x{}, {}", rd, rs1, immediate)
            }
            FullInstruction::CSR(csr) => write!(f, "{}", csr),
            FullInstruction::MulOp(op) => write!(f, "{}", op),
            FullInstruction::Atomic(atomic) => write!(f, "{}", atomic),
            FullInstruction::FPLoad(load) => write!(f, "{}", load),
            FullInstruction::FPStore(store) => write!(f, "{}", store),
            FullInstruction::FPFusedMultiplyOp(op) => write!(f, "{}", op),
            FullInstruction::FPSingleOp(op) => write!(f, "{}", op),
            FullInstruction::FPDoubleOp(op) => write!(f, "{}", op),
        }
    }
}

// Perform sign extension on a u16.
#[inline]
pub fn sign_extend_u16(n: u16, b: u8) -> i16 {
    let shift = 16 - b;
    (n << shift) as i16 >> shift
}

// A compressed RISC-V / RV32C instruction.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum CompressedInstruction {
    ADDI4SPN,
    // if bool is true, this was an SP instruction
    Load(Load, bool),
    Store(Store, bool),
    FPLoad(FPLoad, bool),
    FPStore(FPStore, bool),

    ADDI,
    JAL,
    LI,
    ADDI16SP(i16),
    LUI(usize, u32),
    NOP,
    IMMShift(IMMShift),
    ANDI,
    SUB,
    XOR,
    OR,
    AND,
    SUBW,
    ADDW,
    J,
    BEQZ,
    BNEZ,

    JR(usize, usize),
    MV(usize, usize),
    JALR,
    ADD,
    EBREAK,

    HINT,
    NSE,
    RES,
}

impl CompressedInstruction {
    // Decode a compressed RISC-V / RV32C instruction.
    pub fn decode(inst: u16) -> CompressedInstruction {
        if inst == 0 {
            panic!("illegal instruction");
        }

        let opcode = inst & 0b11;
        let r1 = ((inst >> 2) & 0b11111) as usize;
        let r2 = ((inst >> 7) & 0b11111) as usize;
        let funct3 = inst >> 13;

        match opcode {
            0 => match funct3 {
                0b000 => CompressedInstruction::ADDI4SPN,
                0b001 => {
                    let imm = (((inst >> 5) & 0b11) << 3) | ((inst >> 10) & 0b111);
                    let offset = imm as i32 * 8;
                    CompressedInstruction::FPLoad(
                        FPLoad {
                            rd: r1 & 0b111,
                            rs1: r2 & 0b111,
                            offset,
                            precision: FPPrecision::Double,
                        },
                        false,
                    )
                }
                0b010 => {
                    let imm = (((inst >> 5) & 1) << 4)
                        | (((inst >> 10) & 0b111) << 1)
                        | ((inst >> 6) & 1);
                    let offset = imm as i32 * 4;
                    CompressedInstruction::Load(
                        Load {
                            rd: r1 & 0b111,
                            rs1: r2 & 0b111,
                            offset,
                            mode: LoadMode::Word,
                        },
                        false,
                    )
                }
                0b011 => {
                    let imm = (((inst >> 5) & 1) << 4)
                        | (((inst >> 10) & 0b111) << 1)
                        | ((inst >> 6) & 1);
                    let offset = imm as i32 * 4;
                    CompressedInstruction::FPLoad(
                        FPLoad {
                            rd: r1 & 0b111,
                            rs1: r2 & 0b111,
                            offset,
                            precision: FPPrecision::Single,
                        },
                        false,
                    )
                }
                0b101 => {
                    let imm = (((inst >> 5) & 0b11) << 3) | ((inst >> 10) & 0b111);
                    let offset = imm as i32 * 8;
                    CompressedInstruction::FPStore(
                        FPStore {
                            rs1: r2 & 0b111,
                            rs2: r1 & 0b111,
                            offset,
                            precision: FPPrecision::Double,
                        },
                        false,
                    )
                }
                0b110 => {
                    let imm = (((inst >> 5) & 1) << 4)
                        | (((inst >> 10) & 0b111) << 1)
                        | ((inst >> 6) & 1);
                    let offset = imm as i32 * 4;
                    CompressedInstruction::Store(
                        Store {
                            rs1: r1 & 0b111,
                            rs2: r2 & 0b111,
                            offset,
                            mode: StoreMode::Word,
                        },
                        false,
                    )
                }
                0b111 => {
                    let imm = (((inst >> 5) & 1) << 4)
                        | (((inst >> 10) & 0b111) << 1)
                        | ((inst >> 6) & 1);
                    let offset = imm as i32 * 4;
                    CompressedInstruction::FPStore(
                        FPStore {
                            rs1: r2 & 0b111,
                            rs2: r1 & 0b111,
                            offset,
                            precision: FPPrecision::Single,
                        },
                        false,
                    )
                }
                _ => panic!("illegal instruction"),
            },
            1 => match funct3 {
                0b000 => CompressedInstruction::ADDI,
                0b001 => CompressedInstruction::JAL,
                0b010 => CompressedInstruction::LI,
                0b011 => {
                    if r2 == 2 {
                        let imm = ((inst >> 12) & 1) << 5
                            | ((inst >> 3) & 0b11) << 3
                            | ((inst >> 5) & 1) << 2
                            | ((inst >> 2) & 1) << 1
                            | ((inst >> 6) & 1);
                        let offset = sign_extend_u16(imm, 6) * 16;
                        CompressedInstruction::ADDI16SP(offset)
                    } else if r2 != 0 {
                        CompressedInstruction::LUI(r2, 0)
                    } else {
                        CompressedInstruction::NOP
                    }
                }
                0b100 => {
                    let rd = r2 & 0b111;
                    let imm5 = (inst >> 12) & 1;
                    let funct2_major = (inst >> 10) & 0b11;
                    let funct2_minor = (inst >> 5) & 0b11;

                    match funct2_major {
                        0b00 | 0b01 => {
                            let shamt5 = (inst >> 12) & 1;
                            let amount = ((shamt5 << 5) | r1 as u16) as u32;

                            let mode = if funct2_major == 0b00 {
                                ShiftMode::LogicalRight
                            } else {
                                ShiftMode::ArithmeticRight
                            };

                            if shamt5 == 0 && amount != 0 {
                                CompressedInstruction::IMMShift(IMMShift {
                                    rd,
                                    rs1: rd,
                                    amount,
                                    mode,
                                    compressed: true,
                                })
                            } else if amount == 0 {
                                CompressedInstruction::HINT // RV32/64C
                            } else if shamt5 != 0 {
                                CompressedInstruction::NSE // RV32C
                            } else {
                                panic!("illegal instruction")
                            }
                        }
                        0b10 => CompressedInstruction::ANDI,
                        0b11 if imm5 == 0 => match funct2_minor {
                            0b00 => CompressedInstruction::SUB,
                            0b01 => CompressedInstruction::XOR,
                            0b10 => CompressedInstruction::OR,
                            0b11 => CompressedInstruction::AND,
                            _ => panic!("illegal instruction"),
                        },
                        0b11 if imm5 == 1 => match funct2_minor {
                            0b00 => CompressedInstruction::SUBW,
                            0b01 => CompressedInstruction::ADDW,
                            0b10 => CompressedInstruction::RES,
                            0b11 => CompressedInstruction::RES,
                            _ => panic!("illegal instruction"),
                        },
                        _ => panic!("illegal instruction"),
                    }
                }
                0b101 => CompressedInstruction::J,
                0b110 => CompressedInstruction::BEQZ,
                0b111 => CompressedInstruction::BNEZ,
                _ => panic!("illegal instruction"),
            },
            2 => match funct3 {
                0b000 => {
                    let shamt5 = (inst >> 12) & 1;
                    let amount = ((shamt5 << 5) | r1 as u16) as u32;

                    if shamt5 == 0 && amount != 0 {
                        CompressedInstruction::IMMShift(IMMShift {
                            rd: r2,
                            rs1: r2,
                            amount,
                            mode: ShiftMode::LogicalLeft,
                            compressed: true,
                        })
                    } else if amount == 0 {
                        CompressedInstruction::HINT // RV32/64C
                    } else if shamt5 != 0 {
                        CompressedInstruction::NSE
                    } else {
                        CompressedInstruction::HINT // all base ISAs
                    }
                }
                0b001 => {
                    let imm = (((inst >> 7) & 0b111) << 3) | ((inst >> 10) & 0b111);
                    let offset = imm as i32 * 8;
                    CompressedInstruction::FPLoad(
                        FPLoad {
                            rd: r1 & 0b111,
                            rs1: r2 & 0b111,
                            offset,
                            precision: FPPrecision::Double,
                        },
                        true,
                    )
                }
                0b010 => {
                    let imm = (((inst >> 7) & 0b11) << 4) | ((inst >> 9) & 0b1111);
                    let offset = imm as i32 * 4;
                    CompressedInstruction::Load(
                        Load {
                            rd: r1 & 0b111,
                            rs1: r2 & 0b111,
                            offset,
                            mode: LoadMode::Word,
                        },
                        true,
                    )
                }
                0b011 => {
                    let imm = (((inst >> 7) & 0b11) << 4) | ((inst >> 9) & 0b1111);
                    let offset = imm as i32 * 4;
                    CompressedInstruction::FPLoad(
                        FPLoad {
                            rd: r1 & 0b111,
                            rs1: r2 & 0b111,
                            offset,
                            precision: FPPrecision::Single,
                        },
                        true,
                    )
                }
                0b100 => {
                    if (inst >> 12) != 0b1001 {
                        if r2 != 0 {
                            if r1 == 0 {
                                CompressedInstruction::JR(r2, r1)
                            } else {
                                CompressedInstruction::MV(r2, r1)
                            }
                        } else {
                            CompressedInstruction::HINT
                        }
                    } else if r2 != 0 {
                        if r1 == 0 {
                            CompressedInstruction::JALR
                        } else {
                            CompressedInstruction::ADD
                        }
                    } else if r1 == 0 {
                        CompressedInstruction::EBREAK
                    } else {
                        CompressedInstruction::HINT
                    }
                }
                0b101 => {
                    let imm = (((inst >> 7) & 0b111) << 3) | ((inst >> 10) & 0b111);
                    let offset = imm as i32 * 8;
                    // TODO: im stupid
                    CompressedInstruction::FPStore(
                        FPStore {
                            rs1: r2 & 0b111,
                            rs2: r1 & 0b111,
                            offset,
                            precision: FPPrecision::Double,
                        },
                        true,
                    )
                }
                0b110 => {
                    let imm = (((inst >> 7) & 0b11) << 4) | ((inst >> 9) & 0b1111);
                    let offset = imm as i32 * 4;
                    CompressedInstruction::Store(
                        Store {
                            rs1: r1 & 0b111,
                            rs2: r2 & 0b111,
                            offset,
                            mode: StoreMode::Word,
                        },
                        true,
                    )
                }
                0b111 => {
                    let imm = (((inst >> 7) & 0b11) << 4) | ((inst >> 9) & 0b1111);
                    let offset = imm as i32 * 4;
                    CompressedInstruction::FPStore(
                        FPStore {
                            rs1: r2 & 0b111,
                            rs2: r1 & 0b111,
                            offset,
                            precision: FPPrecision::Single,
                        },
                        true,
                    )
                }
                _ => panic!("illegal instruction"),
            },
            _ => panic!("illegal instruction"),
        }
    }
}

impl Display for CompressedInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unimplemented!()
    }
}
