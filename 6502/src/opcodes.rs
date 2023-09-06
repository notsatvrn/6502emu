use phf::{Map, phf_map};

pub static OPCODES: Map<u8, &'static str> = phf_map! {
    0xEAu8 => "nop",
};
