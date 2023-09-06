use libemu6502::opcodes::OPCODES;

fn main() {
    for (hex, op) in OPCODES.entries() {
        println!("Hex: 0x{:X}, Operation: {}", hex, op);
    }
}
