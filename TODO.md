# RISC-V 32-Bit

## RV32I
- LUI: ☑
- AUPIC: ☑
- JAL: ☑
- JALR: ☑
- BEQ: ☑
- BNE: ☑ (untested)
- BLT: ☑ (untested)
- BGE: ☑ (untested)
- BLTU: ☑ (untested)
- BGEU: ☑ (untested)
- LB: ☑
- LH: ☑
- LW: ☑
- LBU: ☑
- LHU: ☑
- SB: ☑
- SH: ☑
- SW: ☑
- ADDI: ☑
- SLTI: ☑ (untested)
- SLTIU: ☑ (untested)
- XORI: ☑ (untested)
- ORI: ☑ (untested)
- ANDI: ☑ (untested)
- SLLI: ☑
- SRLI: ☑
- SRAI: ☑
- ADD: ☑ (untested)
- SUB: ☑ (untested)
- SLL: ☑ (untested)
- SLT: ☑ (untested)
- SLTU: ☑ (untested)
- XOR: ☑ (untested)
- SRL: ☑ (untested)
- SRA: ☑ (untested)
- OR: ☑ (untested)
- AND: ☑ (untested)
- FENCE: ☒
- ECALL: ☒
- EBREAK: ☒

## Zifencei
- FENCE.I: ☒

## Zicsr
- CSRRW: ☒
- CSRRS: ☒
- CSRRC: ☒
- CSRRWI: ☒
- CSRRSI: ☒
- CSRRCI: ☒

## RV32M
- MUL: ☑ (untested)
- MULH: ☑ (untested)
- MULHSU: ☑ (untested)
- MULHU: ☑ (untested)
- DIV: ☑ (untested)
- DIVU: ☑ (untested)
- REM: ☑ (untested)
- REMU: ☑ (untested)

## RV32A
- LR.W: ☒
- SC.W: ☒
- AMOSWAP.W: ☒
- AMOADD.W: ☒
- AMOXOR.W: ☒
- AMOAND.W: ☒
- AMOOR.W: ☒
- AMOMIN.W: ☒
- AMOMAX.W: ☒
- AMOMINU.W: ☒
- AMOMAXU.W: ☒

## RV32F
- FLW: ☑ (untested)
- FSW: ☑ (untested)
- FMADD.S: ☑ (untested)
- FMSUB.S: ☑ (untested)
- FNMSUB.S: ☑ (untested)
- FNMADD.S: ☑ (untested)
- FADD.S: ☑ (untested)
- FSUB.S: ☑ (untested)
- FMUL.S: ☑ (untested)
- FDIV.S: ☑ (untested)
- FSQRT.S: ☑ (untested)
- FSGNJ.S: ☑ (untested)
- FSGNJN.S: ☑ (untested)
- FSGNJX.S: ☑ (untested)
- FMIN.S: ☑ (untested)
- FMAX.S: ☑ (untested)
- FCVT.W.S: ☑ (untested)
- FCVT.WU.S: ☑ (untested)
- FMV.X.W: ☑ (untested)
- FEQ.S: ☑ (untested)
- FLT.S: ☑ (untested)
- FLE.S: ☑ (untested)
- FCLASS.S: ☑ (untested)
- FCVT.S.W: ☑ (untested)
- FCVT.S.WU: ☑ (untested)
- FMV.W.X: ☑ (untested)

## RV32D
- FLD: ☑ (untested)
- FSD: ☑ (untested)
- FMADD.D: ☑ (untested)
- FMSUB.D: ☑ (untested)
- FNMSUB.D: ☑ (untested)
- FNMADD.D: ☑ (untested)
- FADD.D: ☑ (untested)
- FSUB.D: ☑ (untested)
- FMUL.D: ☑ (untested)
- FDIV.D: ☑ (untested)
- FSQRT.D: ☑ (untested)
- FSGNJ.D: ☑ (untested)
- FSGNJN.D: ☑ (untested)
- FSGNJX.D: ☑ (untested)
- FMIN.D: ☑ (untested)
- FMAX.D: ☑ (untested)
- FCVT.S.D: ☑ (untested)
- FCVT.D.S: ☑ (untested)
- FEQ.D: ☑ (untested)
- FLT.D: ☑ (untested)
- FLE.D: ☑ (untested)
- FCLASS.D: ☑ (untested)
- FCVT.W.D: ☑ (untested)
- FCVT.WU.D: ☑ (untested)
- FCVT.D.W: ☑ (untested)
- FCVT.D.WU: ☑ (untested)

## RV32C

### Quadrant 0
- C.ADDI4SPN: ☒
- C.FLD: ☒
- C.LW: ☒
- C.FLW: ☒
- C.FSD: ☒
- C.SW: ☒
- C.FSW: ☒

### Quadrant 1
- C.ADDI: ☒
- C.JAL: ☒
- C.LI: ☒
- C.ADDI16SP: ☒
- C.LUI: ☒
- C.SRLI: ☑ (untested)
- C.SRAI: ☑ (untested)
- C.ANDI: ☒
- C.SUB: ☒
- C.XOR: ☒
- C.OR: ☒
- C.AND: ☒
- C.SUBW: ☒
- C.ADDW: ☒
- C.J: ☒
- C.BEQZ: ☒
- C.BNEZ: ☒

### Quadrant 2
- C.SLLI: ☑ (untested)
- C.FLDSP: ☑ (untested)
- C.LWSP: ☑ (untested)
- C.FLWSP: ☑ (untested)
- C.ADD: ☑ (untested)
- C.MV: ☑ (untested)
- C.JR: ☑ (untested)
- C.JALR: ☑ (untested)
- C.FSDSP: ☑ (untested)
- C.SWSP: ☑ (untested)
- C.FSWSP: ☑ (untested)
