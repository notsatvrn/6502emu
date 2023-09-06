mod opcodes;

use opcodes::OPCODES;

fn main() {
    for &(hex, op) in &OPCODES {
        println!("Hex: 0x{:X}, Operation: {}", hex, op);
    }
}
